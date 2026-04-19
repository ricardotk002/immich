import {
  CopyObjectCommand,
  DeleteObjectCommand,
  DeleteObjectsCommand,
  GetObjectCommand,
  HeadObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from '@aws-sdk/client-s3';
import { Upload } from '@aws-sdk/lib-storage';
import archiver from 'archiver';
import { ChokidarOptions } from 'chokidar';
import { Stats } from 'node:fs';
import { ReadOptionsWithBuffer } from 'node:fs';
import { PassThrough, Readable, Writable } from 'node:stream';
import { createGunzip, createGzip } from 'node:zlib';
import { CrawlOptionsDto, WalkOptionsDto } from 'src/dtos/library.dto';
import { LoggingRepository } from 'src/repositories/logging.repository';
import {
  DiskUsage,
  ImmichReadStream,
  ImmichZipStream,
  IStorageRepository,
  WatchEvents,
} from 'src/repositories/storage.repository';
import { mimeTypes } from 'src/utils/mime-types';

export interface MinioConfig {
  endpoint: string;
  port: number;
  accessKey: string;
  secretKey: string;
  bucket: string;
  useSSL: boolean;
}

export class MinioStorageRepository implements IStorageRepository {
  private readonly s3: S3Client;
  private readonly bucket: string;

  constructor(
    config: MinioConfig,
    private logger: LoggingRepository,
  ) {
    this.bucket = config.bucket;
    const protocol = config.useSSL ? 'https' : 'http';
    this.s3 = new S3Client({
      endpoint: `${protocol}://${config.endpoint}:${config.port}`,
      region: 'us-east-1',
      credentials: { accessKeyId: config.accessKey, secretAccessKey: config.secretKey },
      forcePathStyle: true,
    });
  }

  realpath(filepath: string): Promise<string> {
    return Promise.resolve(filepath);
  }

  async readdir(prefix: string): Promise<string[]> {
    const normalizedPrefix = prefix.endsWith('/') ? prefix : `${prefix}/`;
    const keys: string[] = [];
    let continuationToken: string | undefined;

    do {
      const response = await this.s3.send(
        new ListObjectsV2Command({
          Bucket: this.bucket,
          Prefix: normalizedPrefix,
          Delimiter: '/',
          ContinuationToken: continuationToken,
        }),
      );

      for (const obj of response.Contents ?? []) {
        if (obj.Key) {
          keys.push(obj.Key.slice(normalizedPrefix.length));
        }
      }

      continuationToken = response.NextContinuationToken;
    } while (continuationToken);

    return keys;
  }

  async copyFile(source: string, target: string): Promise<void> {
    await this.s3.send(
      new CopyObjectCommand({
        Bucket: this.bucket,
        CopySource: `${this.bucket}/${source}`,
        Key: target,
      }),
    );
  }

  async stat(key: string): Promise<Stats> {
    const response = await this.s3.send(new HeadObjectCommand({ Bucket: this.bucket, Key: key }));
    const mtime = response.LastModified ?? new Date();
    return {
      size: response.ContentLength ?? 0,
      atime: mtime,
      mtime,
      ctime: mtime,
      birthtime: mtime,
      isFile: () => true,
      isDirectory: () => false,
      isSymbolicLink: () => false,
      isBlockDevice: () => false,
      isCharacterDevice: () => false,
      isFIFO: () => false,
      isSocket: () => false,
      dev: 0,
      ino: 0,
      mode: 0o644,
      nlink: 1,
      uid: 0,
      gid: 0,
      rdev: 0,
      blksize: 4096,
      blocks: Math.ceil((response.ContentLength ?? 0) / 512),
      atimeMs: mtime.getTime(),
      mtimeMs: mtime.getTime(),
      ctimeMs: mtime.getTime(),
      birthtimeMs: mtime.getTime(),
    } as unknown as Stats;
  }

  async createFile(key: string, buffer: Buffer): Promise<void> {
    const exists = await this.checkFileExists(key);
    if (exists) {
      const error = new Error(`Object already exists: ${key}`) as NodeJS.ErrnoException;
      error.code = 'EEXIST';
      throw error;
    }
    await this.s3.send(new PutObjectCommand({ Bucket: this.bucket, Key: key, Body: buffer }));
  }

  createWriteStream(key: string): Writable {
    const pass = new PassThrough();
    const upload = new Upload({
      client: this.s3,
      params: { Bucket: this.bucket, Key: key, Body: pass },
    });

    // Start consuming immediately so the PassThrough never blocks the pipeline
    const uploadPromise = upload.done();
    uploadPromise.catch(() => {});

    const proxy = new Writable({
      write(chunk, _encoding, callback) {
        if (!pass.write(chunk)) {
          pass.once('drain', callback);
        } else {
          process.nextTick(callback);
        }
      },
      final(callback) {
        pass.end();
        uploadPromise.then(() => callback()).catch(callback);
      },
    });

    return proxy;
  }

  async createOrOverwriteFile(key: string, buffer: Buffer): Promise<void> {
    await this.s3.send(new PutObjectCommand({ Bucket: this.bucket, Key: key, Body: buffer }));
  }

  async overwriteFile(key: string, buffer: Buffer): Promise<void> {
    const exists = await this.checkFileExists(key);
    if (!exists) {
      const error = new Error(`Object does not exist: ${key}`) as NodeJS.ErrnoException;
      error.code = 'ENOENT';
      throw error;
    }
    await this.s3.send(new PutObjectCommand({ Bucket: this.bucket, Key: key, Body: buffer }));
  }

  async rename(source: string, target: string): Promise<void> {
    await this.copyFile(source, target);
    await this.unlink(source);
  }

  utimes(_key: string, _atime: Date, _mtime: Date): Promise<void> {
    return Promise.resolve();
  }

  createZipStream(): ImmichZipStream {
    const archive = archiver('zip', { store: true });

    const addFile = (key: string, filename: string) => {
      const pass = new PassThrough();
      this.s3
        .send(new GetObjectCommand({ Bucket: this.bucket, Key: key }))
        .then(({ Body }) => {
          if (Body instanceof Readable) {
            Body.pipe(pass);
          } else {
            pass.destroy(new Error('Unexpected MinIO body type'));
          }
        })
        .catch((err) => pass.destroy(err));
      archive.append(pass, { name: filename });
    };

    const finalize = () => archive.finalize();

    return { stream: archive, addFile, finalize };
  }

  createGzip(): PassThrough {
    return createGzip();
  }

  createGunzip(): PassThrough {
    return createGunzip();
  }

  createPlainReadStream(key: string): Readable {
    const pass = new PassThrough();
    this.s3
      .send(new GetObjectCommand({ Bucket: this.bucket, Key: key }))
      .then(({ Body }) => {
        if (Body instanceof Readable) {
          Body.pipe(pass);
        } else {
          pass.destroy(new Error('Unexpected MinIO body type'));
        }
      })
      .catch((err) => pass.destroy(err));
    return pass;
  }

  async createReadStream(key: string, mimeType?: string | null): Promise<ImmichReadStream> {
    const response = await this.s3.send(new GetObjectCommand({ Bucket: this.bucket, Key: key }));
    return {
      stream: response.Body as Readable,
      length: response.ContentLength,
      type: mimeType || response.ContentType || undefined,
    };
  }

  async readFile(key: string, _options?: ReadOptionsWithBuffer<Buffer>): Promise<Buffer> {
    const response = await this.s3.send(new GetObjectCommand({ Bucket: this.bucket, Key: key }));
    const chunks: Buffer[] = [];
    for await (const chunk of response.Body as AsyncIterable<Uint8Array>) {
      chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
  }

  async readTextFile(key: string): Promise<string> {
    const buffer = await this.readFile(key);
    return buffer.toString('utf8');
  }

  async checkFileExists(key: string, _mode?: number): Promise<boolean> {
    try {
      await this.s3.send(new HeadObjectCommand({ Bucket: this.bucket, Key: key }));
      return true;
    } catch (error) {
      if (this.isNotFound(error)) {
        return false;
      }
      throw error;
    }
  }

  async unlink(key: string): Promise<void> {
    try {
      await this.s3.send(new DeleteObjectCommand({ Bucket: this.bucket, Key: key }));
    } catch (error) {
      if (this.isNotFound(error)) {
        this.logger.warn(`Object ${key} does not exist.`);
      } else {
        throw error;
      }
    }
  }

  async unlinkDir(prefix: string, _options: { recursive?: boolean; force?: boolean }): Promise<void> {
    const normalizedPrefix = prefix.endsWith('/') ? prefix : `${prefix}/`;
    let continuationToken: string | undefined;

    do {
      const response = await this.s3.send(
        new ListObjectsV2Command({ Bucket: this.bucket, Prefix: normalizedPrefix, ContinuationToken: continuationToken }),
      );

      const objects = (response.Contents ?? []).filter((o) => o.Key).map((o) => ({ Key: o.Key! }));

      if (objects.length > 0) {
        await this.s3.send(new DeleteObjectsCommand({ Bucket: this.bucket, Delete: { Objects: objects } }));
      }

      continuationToken = response.NextContinuationToken;
    } while (continuationToken);
  }

  removeEmptyDirs(_directory: string, _self?: boolean): Promise<void> {
    return Promise.resolve();
  }

  mkdirSync(_filepath: string): void {
    // no-op: MinIO has no directories
  }

  existsSync(_filepath: string): boolean {
    return true;
  }

  checkDiskUsage(_folder: string): Promise<DiskUsage> {
    return Promise.resolve({ available: Number.MAX_SAFE_INTEGER, free: Number.MAX_SAFE_INTEGER, total: Number.MAX_SAFE_INTEGER });
  }

  async crawl(crawlOptions: CrawlOptionsDto): Promise<string[]> {
    const { pathsToCrawl } = crawlOptions;
    if (pathsToCrawl.length === 0) {
      return [];
    }

    const supportedExtensions = new Set(mimeTypes.getSupportedFileExtensions());
    const results: string[] = [];

    for (const prefix of pathsToCrawl) {
      const normalizedPrefix = prefix.endsWith('/') ? prefix : `${prefix}/`;
      let continuationToken: string | undefined;

      do {
        const response = await this.s3.send(
          new ListObjectsV2Command({
            Bucket: this.bucket,
            Prefix: normalizedPrefix,
            ContinuationToken: continuationToken,
          }),
        );

        for (const obj of response.Contents ?? []) {
          if (!obj.Key) continue;
          const ext = obj.Key.slice(obj.Key.lastIndexOf('.')).toLowerCase();
          if (supportedExtensions.has(ext)) {
            results.push(obj.Key);
          }
        }

        continuationToken = response.NextContinuationToken;
      } while (continuationToken);
    }

    return results;
  }

  async *walk(walkOptions: WalkOptionsDto): AsyncGenerator<string[]> {
    const { pathsToCrawl, take = 1000 } = walkOptions;
    if (pathsToCrawl.length === 0) {
      return;
    }

    const supportedExtensions = new Set(mimeTypes.getSupportedFileExtensions());

    for (const prefix of pathsToCrawl) {
      const normalizedPrefix = prefix.endsWith('/') ? prefix : `${prefix}/`;
      let continuationToken: string | undefined;
      let batch: string[] = [];

      do {
        const response = await this.s3.send(
          new ListObjectsV2Command({
            Bucket: this.bucket,
            Prefix: normalizedPrefix,
            ContinuationToken: continuationToken,
          }),
        );

        for (const obj of response.Contents ?? []) {
          if (!obj.Key) continue;
          const ext = obj.Key.slice(obj.Key.lastIndexOf('.')).toLowerCase();
          if (!supportedExtensions.has(ext)) continue;

          batch.push(obj.Key);
          if (batch.length === take) {
            yield batch;
            batch = [];
          }
        }

        continuationToken = response.NextContinuationToken;
      } while (continuationToken);

      if (batch.length > 0) {
        yield batch;
      }
    }
  }

  watch(_paths: string[], _options: ChokidarOptions, _events: Partial<WatchEvents>): () => Promise<void> {
    return () => Promise.resolve();
  }

  private isNotFound(error: unknown): boolean {
    return (
      error !== null &&
      typeof error === 'object' &&
      '$metadata' in error &&
      (error as { $metadata: { httpStatusCode?: number } }).$metadata?.httpStatusCode === 404
    );
  }
}
