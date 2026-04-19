import { Injectable } from '@nestjs/common';
import { ChokidarOptions } from 'chokidar';
import { Stats } from 'node:fs';
import { ReadOptionsWithBuffer } from 'node:fs';
import { PassThrough, Readable, Writable } from 'node:stream';
import { CrawlOptionsDto, WalkOptionsDto } from 'src/dtos/library.dto';
import { ConfigRepository } from 'src/repositories/config.repository';
import { LoggingRepository } from 'src/repositories/logging.repository';
import { DiskStorageRepository } from 'src/repositories/storage-disk.repository';
import { MinioStorageRepository } from 'src/repositories/storage-minio.repository';

export interface WatchEvents {
  onReady(): void;
  onAdd(path: string): void;
  onChange(path: string): void;
  onUnlink(path: string): void;
  onError(error: Error): void;
}

export interface ImmichReadStream {
  stream: Readable;
  type?: string;
  length?: number;
}

export interface ImmichZipStream extends ImmichReadStream {
  addFile: (inputPath: string, filename: string) => void;
  finalize: () => Promise<void>;
}

export interface DiskUsage {
  available: number;
  free: number;
  total: number;
}

export interface IStorageRepository {
  realpath(filepath: string): Promise<string>;
  readdir(folder: string): Promise<string[]>;
  copyFile(source: string, target: string): Promise<void>;
  stat(filepath: string): Promise<Stats>;
  createFile(filepath: string, buffer: Buffer): Promise<void>;
  createWriteStream(filepath: string): Writable;
  createOrOverwriteFile(filepath: string, buffer: Buffer): Promise<void>;
  overwriteFile(filepath: string, buffer: Buffer): Promise<void>;
  rename(source: string, target: string): Promise<void>;
  utimes(filepath: string, atime: Date, mtime: Date): Promise<void>;
  createZipStream(): ImmichZipStream;
  createGzip(): PassThrough;
  createGunzip(): PassThrough;
  createPlainReadStream(filepath: string): Readable;
  createReadStream(filepath: string, mimeType?: string | null): Promise<ImmichReadStream>;
  readFile(filepath: string, options?: ReadOptionsWithBuffer<Buffer>): Promise<Buffer>;
  readTextFile(filepath: string): Promise<string>;
  checkFileExists(filepath: string, mode?: number): Promise<boolean>;
  unlink(file: string): Promise<void>;
  unlinkDir(folder: string, options: { recursive?: boolean; force?: boolean }): Promise<void>;
  removeEmptyDirs(directory: string, self?: boolean): Promise<void>;
  mkdirSync(filepath: string): void;
  existsSync(filepath: string): boolean;
  checkDiskUsage(folder: string): Promise<DiskUsage>;
  crawl(crawlOptions: CrawlOptionsDto): Promise<string[]>;
  walk(walkOptions: WalkOptionsDto): AsyncGenerator<string[]>;
  watch(paths: string[], options: ChokidarOptions, events: Partial<WatchEvents>): () => Promise<void>;
}

@Injectable()
export class StorageRepository implements IStorageRepository {
  private readonly backend: IStorageRepository;

  constructor(
    private logger: LoggingRepository,
    configRepository: ConfigRepository,
  ) {
    const env = configRepository.getEnv();
    this.backend =
      env.storage.backend === 'minio'
        ? new MinioStorageRepository(env.storage.minio, logger)
        : new DiskStorageRepository(logger);
  }

  realpath(filepath: string) {
    return this.backend.realpath(filepath);
  }

  readdir(folder: string) {
    return this.backend.readdir(folder);
  }

  copyFile(source: string, target: string) {
    return this.backend.copyFile(source, target);
  }

  stat(filepath: string) {
    return this.backend.stat(filepath);
  }

  createFile(filepath: string, buffer: Buffer) {
    return this.backend.createFile(filepath, buffer);
  }

  createWriteStream(filepath: string) {
    return this.backend.createWriteStream(filepath);
  }

  createOrOverwriteFile(filepath: string, buffer: Buffer) {
    return this.backend.createOrOverwriteFile(filepath, buffer);
  }

  overwriteFile(filepath: string, buffer: Buffer) {
    return this.backend.overwriteFile(filepath, buffer);
  }

  rename(source: string, target: string) {
    return this.backend.rename(source, target);
  }

  utimes(filepath: string, atime: Date, mtime: Date) {
    return this.backend.utimes(filepath, atime, mtime);
  }

  createZipStream() {
    return this.backend.createZipStream();
  }

  createGzip() {
    return this.backend.createGzip();
  }

  createGunzip() {
    return this.backend.createGunzip();
  }

  createPlainReadStream(filepath: string) {
    return this.backend.createPlainReadStream(filepath);
  }

  createReadStream(filepath: string, mimeType?: string | null) {
    return this.backend.createReadStream(filepath, mimeType);
  }

  readFile(filepath: string, options?: ReadOptionsWithBuffer<Buffer>) {
    return this.backend.readFile(filepath, options);
  }

  readTextFile(filepath: string) {
    return this.backend.readTextFile(filepath);
  }

  checkFileExists(filepath: string, mode?: number) {
    return this.backend.checkFileExists(filepath, mode);
  }

  unlink(file: string) {
    return this.backend.unlink(file);
  }

  unlinkDir(folder: string, options: { recursive?: boolean; force?: boolean }) {
    return this.backend.unlinkDir(folder, options);
  }

  removeEmptyDirs(directory: string, self?: boolean) {
    return this.backend.removeEmptyDirs(directory, self);
  }

  mkdirSync(filepath: string) {
    return this.backend.mkdirSync(filepath);
  }

  existsSync(filepath: string) {
    return this.backend.existsSync(filepath);
  }

  checkDiskUsage(folder: string) {
    return this.backend.checkDiskUsage(folder);
  }

  crawl(crawlOptions: CrawlOptionsDto) {
    return this.backend.crawl(crawlOptions);
  }

  walk(walkOptions: WalkOptionsDto) {
    return this.backend.walk(walkOptions);
  }

  watch(paths: string[], options: ChokidarOptions, events: Partial<WatchEvents>) {
    return this.backend.watch(paths, options, events);
  }
}
