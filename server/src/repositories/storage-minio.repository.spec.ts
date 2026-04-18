import {
  DeleteObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
  PutObjectCommand,
  S3Client,
} from '@aws-sdk/client-s3';
import { mockClient } from 'aws-sdk-client-mock';
import { Readable } from 'node:stream';
import { LoggingRepository } from 'src/repositories/logging.repository';
import { MinioStorageRepository } from 'src/repositories/storage-minio.repository';
import { automock } from 'test/utils';

const s3Mock = mockClient(S3Client);

const makeLogger = () => automock(LoggingRepository, { args: [, { getEnv: () => ({}) }], strict: false });

const minioConfig = {
  endpoint: 'localhost',
  port: 9000,
  accessKey: 'test',
  secretKey: 'test',
  bucket: 'immich',
  useSSL: false,
};

describe(MinioStorageRepository.name, () => {
  let sut: MinioStorageRepository;

  beforeEach(() => {
    s3Mock.reset();
    sut = new MinioStorageRepository(minioConfig, makeLogger());
  });

  describe('checkFileExists', () => {
    it('returns true when object exists', async () => {
      s3Mock.on(HeadObjectCommand).resolves({});
      await expect(sut.checkFileExists('upload/user/ab/cd/file.jpg')).resolves.toBe(true);
    });

    it('returns false when object does not exist', async () => {
      s3Mock.on(HeadObjectCommand).rejects({ $metadata: { httpStatusCode: 404 }, name: 'NotFound' });
      await expect(sut.checkFileExists('upload/user/ab/cd/file.jpg')).resolves.toBe(false);
    });
  });

  describe('createOrOverwriteFile', () => {
    it('puts object to MinIO', async () => {
      s3Mock.on(PutObjectCommand).resolves({});
      await expect(sut.createOrOverwriteFile('upload/user/ab/cd/file.jpg', Buffer.from('data'))).resolves.toBeUndefined();
    });
  });

  describe('unlink', () => {
    it('deletes object from MinIO', async () => {
      s3Mock.on(DeleteObjectCommand).resolves({});
      await expect(sut.unlink('upload/user/ab/cd/file.jpg')).resolves.toBeUndefined();
    });

    it('warns instead of throwing when object does not exist', async () => {
      s3Mock.on(DeleteObjectCommand).rejects({ $metadata: { httpStatusCode: 404 }, name: 'NotFound' });
      await expect(sut.unlink('upload/user/ab/cd/missing.jpg')).resolves.toBeUndefined();
    });
  });

  describe('readFile', () => {
    it('returns buffer from object body', async () => {
      const body = Readable.from([Buffer.from('hello world')]);
      s3Mock.on(GetObjectCommand).resolves({ Body: body } as any);
      const result = await sut.readFile('upload/user/ab/cd/file.txt');
      expect(result.toString()).toBe('hello world');
    });
  });

  describe('rename', () => {
    it('copies then deletes the source object', async () => {
      s3Mock.on(PutObjectCommand).resolves({});
      s3Mock.on(DeleteObjectCommand).resolves({});
      // rename is implemented as copy + delete
      await expect(sut.rename('old/path.jpg', 'new/path.jpg')).resolves.toBeUndefined();
    });
  });

  describe('existsSync', () => {
    it('always returns true for MinIO (no directory concept)', () => {
      expect(sut.existsSync('any/path')).toBe(true);
    });
  });

  describe('mkdirSync', () => {
    it('is a no-op', () => {
      expect(() => sut.mkdirSync('any/path')).not.toThrow();
    });
  });

  describe('removeEmptyDirs', () => {
    it('is a no-op', async () => {
      await expect(sut.removeEmptyDirs('any/path')).resolves.toBeUndefined();
    });
  });

  describe('utimes', () => {
    it('is a no-op', async () => {
      await expect(sut.utimes('any/path', new Date(), new Date())).resolves.toBeUndefined();
    });
  });

  describe('checkDiskUsage', () => {
    it('returns unlimited disk usage', async () => {
      const result = await sut.checkDiskUsage('any/path');
      expect(result.available).toBe(Number.MAX_SAFE_INTEGER);
      expect(result.free).toBe(Number.MAX_SAFE_INTEGER);
      expect(result.total).toBe(Number.MAX_SAFE_INTEGER);
    });
  });
});
