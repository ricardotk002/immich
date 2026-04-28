import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { BadRequestException, Inject, Injectable, InternalServerErrorException, NotFoundException } from '@nestjs/common';
import { extname } from 'node:path';
import sanitize from 'sanitize-filename';
import sharp from 'sharp';
import { StorageCore } from 'src/cores/storage.core';
import { AuthSharedLink } from 'src/database';
import {
  AssetBulkUploadCheckResponseDto,
  AssetMediaResponseDto,
  AssetMediaStatus,
  AssetRejectReason,
  AssetUploadAction,
} from 'src/dtos/asset-media-response.dto';
import {
  AssetBulkUploadCheckDto,
  AssetMediaCreateDto,
  AssetMediaOptionsDto,
  AssetMediaSize,
  StickerDto,
  StickerResolveDto,
  UploadFieldName,
} from 'src/dtos/asset-media.dto';
import { StickerGenerationRepository } from 'src/repositories/sticker-generation.repository';
import { AssetDownloadOriginalDto } from 'src/dtos/asset.dto';
import { AuthDto } from 'src/dtos/auth.dto';
import {
  AssetFileType,
  AssetVisibility,
  CacheControl,
  ChecksumAlgorithm,
  JobName,
  Permission,
  StorageFolder,
} from 'src/enum';
import { AuthRequest } from 'src/middleware/auth.guard';
import { BaseService } from 'src/services/base.service';
import { UploadFile, UploadRequest } from 'src/types';
import { requireUploadAccess } from 'src/utils/access';
import { asUploadRequest, onBeforeLink } from 'src/utils/asset.util';
import { isAssetChecksumConstraint } from 'src/utils/database';
import { getFilenameExtension, getFileNameWithoutExtension, ImmichFileResponse } from 'src/utils/file';
import { mimeTypes } from 'src/utils/mime-types';
import { fromChecksum } from 'src/utils/request';

export interface AssetMediaRedirectResponse {
  targetSize: AssetMediaSize | 'original';
}

@Injectable()
export class AssetMediaService extends BaseService {
  @Inject()
  private stickerGenerationRepository!: StickerGenerationRepository;

  private get stickerS3(): S3Client {
    return new S3Client({
      endpoint: process.env.STICKER_S3_ENDPOINT ?? 'https://chi.tacc.chameleoncloud.org:7480',
      region: 'us-east-1',
      credentials: {
        accessKeyId: process.env.STICKER_S3_ACCESS_KEY ?? '',
        secretAccessKey: process.env.STICKER_S3_SECRET_KEY ?? '',
      },
      forcePathStyle: true,
    });
  }

  private get stickerBucket(): string {
    return process.env.STICKER_S3_BUCKET ?? 'objstore-proj28';
  }

  async getUploadAssetIdByChecksum(auth: AuthDto, checksum?: string): Promise<AssetMediaResponseDto | undefined> {
    if (!checksum) {
      return;
    }

    const assetId = await this.assetRepository.getUploadAssetIdByChecksum(auth.user.id, fromChecksum(checksum));
    if (!assetId) {
      return;
    }

    return { id: assetId, status: AssetMediaStatus.DUPLICATE };
  }

  canUploadFile({ auth, fieldName, file, body }: UploadRequest): true {
    requireUploadAccess(auth);

    const filename = body.filename || file.originalName;

    switch (fieldName) {
      case UploadFieldName.ASSET_DATA: {
        if (mimeTypes.isAsset(filename)) {
          return true;
        }
        break;
      }

      case UploadFieldName.SIDECAR_DATA: {
        if (mimeTypes.isSidecar(filename)) {
          return true;
        }
        break;
      }

      case UploadFieldName.PROFILE_DATA: {
        if (mimeTypes.isProfile(filename)) {
          return true;
        }
        break;
      }
    }

    this.logger.error(`Unsupported file type ${filename}`);
    throw new BadRequestException(`Unsupported file type ${filename}`);
  }

  getUploadFilename({ auth, fieldName, file, body }: UploadRequest): string {
    requireUploadAccess(auth);

    const extension = extname(body.filename || file.originalName);

    const lookup = {
      [UploadFieldName.ASSET_DATA]: extension,
      [UploadFieldName.SIDECAR_DATA]: '.xmp',
      [UploadFieldName.PROFILE_DATA]: extension,
    };

    return sanitize(`${file.uuid}${lookup[fieldName]}`);
  }

  getUploadFolder({ auth, fieldName, file }: UploadRequest): string {
    auth = requireUploadAccess(auth);

    let folder = StorageCore.getNestedFolder(StorageFolder.Upload, auth.user.id, file.uuid);
    if (fieldName === UploadFieldName.PROFILE_DATA) {
      folder = StorageCore.getFolderLocation(StorageFolder.Profile, auth.user.id);
    }

    this.storageRepository.mkdirSync(folder);

    return folder;
  }

  async onUploadError(request: AuthRequest, file: Express.Multer.File) {
    const uploadFilename = this.getUploadFilename(asUploadRequest(request, file));
    const uploadFolder = this.getUploadFolder(asUploadRequest(request, file));
    const uploadPath = `${uploadFolder}/${uploadFilename}`;

    await this.jobRepository.queue({ name: JobName.FileDelete, data: { files: [uploadPath] } });
  }

  async uploadAsset(
    auth: AuthDto,
    dto: AssetMediaCreateDto,
    file: UploadFile,
    sidecarFile?: UploadFile,
  ): Promise<AssetMediaResponseDto> {
    try {
      await this.requireAccess({
        auth,
        permission: Permission.AssetUpload,
        // do not need an id here, but the interface requires it
        ids: [auth.user.id],
      });

      this.requireQuota(auth, file.size);

      if (dto.livePhotoVideoId) {
        await onBeforeLink(
          { asset: this.assetRepository, event: this.eventRepository },
          { userId: auth.user.id, livePhotoVideoId: dto.livePhotoVideoId },
        );
      }
      const asset = await this.create(auth.user.id, dto, file, sidecarFile);

      if (auth.sharedLink) {
        await this.addToSharedLink(auth.sharedLink, asset.id);
      }

      await this.userRepository.updateUsage(auth.user.id, file.size);

      return { id: asset.id, status: AssetMediaStatus.CREATED };
    } catch (error: any) {
      return this.handleUploadError(error, auth, file, sidecarFile);
    }
  }

  async downloadOriginal(auth: AuthDto, id: string, dto: AssetDownloadOriginalDto): Promise<ImmichFileResponse> {
    await this.requireAccess({ auth, permission: Permission.AssetDownload, ids: [id] });

    if (auth.sharedLink) {
      dto.edited = true;
    }

    const { originalPath, originalFileName, editedPath } = await this.assetRepository.getForOriginal(
      id,
      dto.edited ?? false,
    );

    const path = editedPath ?? originalPath!;

    return new ImmichFileResponse({
      path,
      fileName: getFileNameWithoutExtension(originalFileName) + getFilenameExtension(path),
      contentType: mimeTypes.lookup(path),
      cacheControl: CacheControl.PrivateWithCache,
    });
  }

  async viewThumbnail(
    auth: AuthDto,
    id: string,
    dto: AssetMediaOptionsDto,
  ): Promise<ImmichFileResponse | AssetMediaRedirectResponse> {
    await this.requireAccess({ auth, permission: Permission.AssetView, ids: [id] });

    if (dto.size === AssetMediaSize.Original) {
      throw new BadRequestException('May not request original file');
    }

    if (auth.sharedLink) {
      dto.edited = true;
    }

    const size = (dto.size ?? AssetMediaSize.THUMBNAIL) as unknown as AssetFileType;
    const { originalPath, originalFileName, path } = await this.assetRepository.getForThumbnail(
      id,
      size,
      dto.edited ?? false,
    );

    if (size === AssetFileType.FullSize && mimeTypes.isWebSupportedImage(originalPath) && !dto.edited) {
      // use original file for web supported images
      return { targetSize: 'original' };
    }

    if (dto.size === AssetMediaSize.FULLSIZE && !path) {
      // downgrade to preview if fullsize is not available.
      // e.g. disabled or not yet (re)generated
      return { targetSize: AssetMediaSize.PREVIEW };
    }

    if (!path) {
      throw new NotFoundException('Asset media not found');
    }

    const fileNameBase =
      auth.sharedLink && !auth.sharedLink.showExif ? id : getFileNameWithoutExtension(originalFileName);
    const fileName = `${fileNameBase}_${size}${getFilenameExtension(path)}`;

    return new ImmichFileResponse({
      fileName,
      path,
      contentType: mimeTypes.lookup(path),
      cacheControl: CacheControl.PrivateWithCache,
    });
  }

  async playbackVideo(auth: AuthDto, id: string): Promise<ImmichFileResponse> {
    await this.requireAccess({ auth, permission: Permission.AssetView, ids: [id] });

    const asset = await this.assetRepository.getForVideo(id);

    if (!asset) {
      throw new NotFoundException('Asset not found or asset is not a video');
    }

    const filepath = asset.encodedVideoPath || asset.originalPath;

    return new ImmichFileResponse({
      path: filepath,
      contentType: mimeTypes.lookup(filepath),
      cacheControl: CacheControl.PrivateWithCache,
    });
  }

  async generateStickerMask(auth: AuthDto, id: string, dto: StickerDto): Promise<{ mask: string; stickerId: string }> {
    await this.requireAccess({ auth, permission: Permission.AssetView, ids: [id] });

    const asset = await this.assetRepository.getById(id, {});
    if (!asset) {
      throw new NotFoundException('Asset not found');
    }

    const imageBuffer = await this.storageRepository.readFile(asset.originalPath);
    const imageBase64 = imageBuffer.toString('base64');

    const inferenceUrl = `${process.env.STICKER_GEN_URL ?? 'http://localhost:8004'}/predict`;
    const body: Record<string, unknown> = { image: imageBase64 };
    if (dto.bbox) {
      body.bbox = dto.bbox;
    }
    if (dto.pointCoords) {
      body.point_coords = dto.pointCoords;
    }

    const t0 = Date.now();
    const response = await fetch(inferenceUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const processingTimeMs = Date.now() - t0;

    if (!response.ok) {
      throw new InternalServerErrorException(`Inference request failed: ${response.statusText}`);
    }

    const result = (await response.json()) as { mask: string };
    const maskBuffer = Buffer.from(result.mask, 'base64');

    let stickerId: string;

    if (dto.existingStickerId) {
      // Retry: update the existing row instead of inserting a new one
      const existing = await this.stickerGenerationRepository.getById(dto.existingStickerId);
      if (!existing || existing.assetId !== id || existing.userId !== auth.user.id) {
        throw new NotFoundException('Sticker not found');
      }
      stickerId = dto.existingStickerId;
      const maskKey = `stickers/${id}/${stickerId}/mask.png`;
      try {
        await this.stickerS3.send(
          new PutObjectCommand({ Bucket: this.stickerBucket, Key: maskKey, Body: maskBuffer, ContentType: 'image/png' }),
        );
      } catch {
        // mask binary persisted in DB below
      }
      await this.stickerGenerationRepository.update(stickerId, {
        bbox: dto.bbox ?? null,
        pointCoords: dto.pointCoords ?? null,
        mlSuggestedMask: maskKey,
        mlSuggestedMaskData: maskBuffer,
        processingTimeMs,
        numTries: (existing.numTries as number) + 1,
      });
    } else {
      // First attempt: insert a new row
      stickerId = await this.stickerGenerationRepository.insert({
        userId: auth.user.id,
        assetId: id,
        bbox: dto.bbox ?? null,
        pointCoords: dto.pointCoords ?? null,
        mlSuggestedMaskData: maskBuffer,
        processingTimeMs,
      });
      const maskKey = `stickers/${id}/${stickerId}/mask.png`;
      try {
        await this.stickerS3.send(
          new PutObjectCommand({ Bucket: this.stickerBucket, Key: maskKey, Body: maskBuffer, ContentType: 'image/png' }),
        );
        await this.stickerGenerationRepository.update(stickerId, { mlSuggestedMask: maskKey });
      } catch {
        // mask binary is already persisted in the DB row
      }
    }

    return { mask: result.mask, stickerId };
  }

  async resolveStickerMask(auth: AuthDto, assetId: string, stickerId: string, dto: StickerResolveDto): Promise<void> {
    await this.requireAccess({ auth, permission: Permission.AssetView, ids: [assetId] });

    const row = await this.stickerGenerationRepository.getById(stickerId);
    if (!row || row.assetId !== assetId || row.userId !== auth.user.id) {
      throw new NotFoundException('Sticker not found');
    }

    if (!dto.saved) {
      await this.stickerGenerationRepository.update(stickerId, {
        saved: false,
        numTries: dto.numTries ?? (row.numTries as number),
        editedPixels: dto.editedPixels ?? 0,
      });
      return;
    }

    // Determine which mask to use for the sticker composite
    const maskBase64 = dto.userSavedMask ?? null;
    let maskBuffer: Buffer;
    if (maskBase64) {
      maskBuffer = Buffer.from(maskBase64, 'base64');
    } else if (row.mlSuggestedMaskData) {
      maskBuffer = row.mlSuggestedMaskData as Buffer;
    } else {
      throw new BadRequestException('No mask available to generate sticker');
    }

    // Load original image and composite with mask to produce a transparent PNG sticker
    const asset = await this.assetRepository.getById(assetId, {});
    if (!asset) {
      throw new NotFoundException('Asset not found');
    }
    const imageBuffer = await this.storageRepository.readFile(asset.originalPath);
    const { width, height } = await sharp(imageBuffer).metadata();
    const { data: greyData } = await sharp(maskBuffer)
      .resize(width, height)
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });
    // Model returns background=white when a bbox prompt was used; invert so foreground is kept
    const invertMask = !!(row.bbox && (row.bbox as number[]).length === 4);
    const alphaRgba = Buffer.alloc(width! * height! * 4, 0);
    for (let i = 0; i < width! * height!; i++) {
      const grey = (greyData as Buffer)[i];
      alphaRgba[i * 4 + 3] = invertMask ? 255 - grey : grey;
    }
    const stickerBuffer = await sharp(imageBuffer)
      .ensureAlpha()
      .composite([{ input: alphaRgba, raw: { width: width!, height: height!, channels: 4 }, blend: 'dest-in' }])
      .png()
      .toBuffer();

    const stickerKey = `stickers/${assetId}/${stickerId}/sticker.png`;
    await this.stickerS3.send(
      new PutObjectCommand({ Bucket: this.stickerBucket, Key: stickerKey, Body: stickerBuffer, ContentType: 'image/png' }),
    );

    await this.stickerGenerationRepository.update(stickerId, {
      saved: true,
      numTries: dto.numTries ?? (row.numTries as number),
      editedPixels: dto.editedPixels ?? 0,
      userSavedMask: dto.userSavedMask ?? null,
      s3StickerKey: stickerKey,
    });
  }

  async bulkUploadCheck(auth: AuthDto, dto: AssetBulkUploadCheckDto): Promise<AssetBulkUploadCheckResponseDto> {
    const checksums: Buffer[] = dto.assets.map((asset) => fromChecksum(asset.checksum));
    const results = await this.assetRepository.getByChecksums(auth.user.id, checksums);
    const checksumMap: Record<string, { id: string; isTrashed: boolean }> = {};

    for (const { id, deletedAt, checksum } of results) {
      checksumMap[checksum.toString('hex')] = { id, isTrashed: !!deletedAt };
    }

    return {
      results: dto.assets.map(({ id, checksum }) => {
        const duplicate = checksumMap[fromChecksum(checksum).toString('hex')];
        if (duplicate) {
          return {
            id,
            action: AssetUploadAction.REJECT,
            reason: AssetRejectReason.DUPLICATE,
            assetId: duplicate.id,
            isTrashed: duplicate.isTrashed,
          };
        }

        return {
          id,
          action: AssetUploadAction.ACCEPT,
        };
      }),
    };
  }

  private async addToSharedLink(sharedLink: AuthSharedLink, assetId: string) {
    await (sharedLink.albumId
      ? this.albumRepository.addAssetIds(sharedLink.albumId, [assetId])
      : this.sharedLinkRepository.addAssets(sharedLink.id, [assetId]));
  }

  private async handleUploadError(
    error: any,
    auth: AuthDto,
    file: UploadFile,
    sidecarFile?: UploadFile,
  ): Promise<AssetMediaResponseDto> {
    // clean up files
    await this.jobRepository.queue({
      name: JobName.FileDelete,
      data: { files: [file.originalPath, sidecarFile?.originalPath] },
    });

    // handle duplicates with a success response
    if (isAssetChecksumConstraint(error)) {
      const duplicateId = await this.assetRepository.getUploadAssetIdByChecksum(auth.user.id, file.checksum);
      if (!duplicateId) {
        this.logger.error(`Error locating duplicate for checksum constraint`);
        throw new InternalServerErrorException();
      }

      if (auth.sharedLink) {
        await this.addToSharedLink(auth.sharedLink, duplicateId);
      }

      this.logger.debug(`Duplicate asset upload rejected: existing asset ${duplicateId}`);
      return { status: AssetMediaStatus.DUPLICATE, id: duplicateId };
    }

    this.logger.error(`Error uploading file ${error}`, error?.stack);
    throw error;
  }

  private async create(ownerId: string, dto: AssetMediaCreateDto, file: UploadFile, sidecarFile?: UploadFile) {
    const asset = await this.assetRepository.create({
      ownerId,
      libraryId: null,

      checksum: file.checksum,
      checksumAlgorithm: ChecksumAlgorithm.sha1File,
      originalPath: file.originalPath,

      fileCreatedAt: dto.fileCreatedAt,
      fileModifiedAt: dto.fileModifiedAt,
      localDateTime: dto.fileCreatedAt,

      type: mimeTypes.assetType(file.originalPath),
      isFavorite: dto.isFavorite,
      duration: dto.duration || null,
      visibility: dto.visibility ?? AssetVisibility.Timeline,
      livePhotoVideoId: dto.livePhotoVideoId,
      originalFileName: dto.filename || file.originalName,
    });

    if (dto.metadata?.length) {
      await this.assetRepository.upsertMetadata(asset.id, dto.metadata);
    }

    if (sidecarFile) {
      await this.assetRepository.upsertFile({
        assetId: asset.id,
        path: sidecarFile.originalPath,
        type: AssetFileType.Sidecar,
      });
      await this.storageRepository.utimes(sidecarFile.originalPath, new Date(), new Date(dto.fileModifiedAt));
    }
    await this.storageRepository.utimes(file.originalPath, new Date(), new Date(dto.fileModifiedAt));
    await this.assetRepository.upsertExif(
      { assetId: asset.id, fileSizeInByte: file.size },
      { lockedPropertiesBehavior: 'override' },
    );

    await this.eventRepository.emit('AssetCreate', { asset });

    await this.jobRepository.queue({ name: JobName.AssetExtractMetadata, data: { id: asset.id, source: 'upload' } });

    return asset;
  }

  private requireQuota(auth: AuthDto, size: number) {
    if (auth.user.quotaSizeInBytes !== null && auth.user.quotaSizeInBytes < auth.user.quotaUsageInBytes + size) {
      throw new BadRequestException('Quota has been exceeded!');
    }
  }
}
