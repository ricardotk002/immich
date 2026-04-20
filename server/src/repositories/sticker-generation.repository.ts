import { Injectable } from '@nestjs/common';
import { InjectKysely } from 'nestjs-kysely';
import { Kysely } from 'kysely';
import { DB } from 'src/schema';
import { StickerQualityStatus } from 'src/schema/tables/sticker-generation.table';

export interface StickerGenerationInsert {
  userId: string;
  assetId: string;
  bbox?: number[] | null;
  pointCoords?: number[][] | null;
  mlSuggestedMask?: string | null;
  mlSuggestedMaskData?: Buffer | null;
  processingTimeMs?: number;
}

export interface StickerGenerationUpdate {
  saved?: boolean;
  numTries?: number;
  editedPixels?: number;
  mlSuggestedMask?: string | null;
  userSavedMask?: string | null;
  qualityStatus?: StickerQualityStatus;
  qualityCheckedAt?: Date | null;
  qualityCheckVersion?: number | null;
  qualityFailReasonsJson?: string | null;
  retrainRunId?: string | null;
  usedForTraining?: boolean;
}

@Injectable()
export class StickerGenerationRepository {
  constructor(@InjectKysely() private db: Kysely<DB>) {}

  async insert(row: StickerGenerationInsert): Promise<string> {
    const result = await this.db
      .insertInto('sticker_generation')
      .values({
        userId: row.userId,
        assetId: row.assetId,
        bbox: row.bbox ?? null,
        pointCoords: row.pointCoords ?? null,
        mlSuggestedMask: row.mlSuggestedMask ?? null,
        mlSuggestedMaskData: row.mlSuggestedMaskData ?? null,
        processingTimeMs: row.processingTimeMs ?? 0,
      })
      .returning('id')
      .executeTakeFirstOrThrow();

    return result.id as string;
  }

  async update(id: string, patch: StickerGenerationUpdate): Promise<void> {
    const values: Record<string, unknown> = {};
    if (patch.saved !== undefined) values.saved = patch.saved;
    if (patch.numTries !== undefined) values.numTries = patch.numTries;
    if (patch.editedPixels !== undefined) values.editedPixels = patch.editedPixels;
    if (patch.mlSuggestedMask !== undefined) values.mlSuggestedMask = patch.mlSuggestedMask;
    if (patch.userSavedMask !== undefined) values.userSavedMask = patch.userSavedMask;
    if (patch.qualityStatus !== undefined) values.qualityStatus = patch.qualityStatus;
    if (patch.qualityCheckedAt !== undefined) values.qualityCheckedAt = patch.qualityCheckedAt;
    if (patch.qualityCheckVersion !== undefined) values.qualityCheckVersion = patch.qualityCheckVersion;
    if (patch.qualityFailReasonsJson !== undefined) values.qualityFailReasonsJson = patch.qualityFailReasonsJson;
    if (patch.retrainRunId !== undefined) values.retrainRunId = patch.retrainRunId;
    if (patch.usedForTraining !== undefined) values.usedForTraining = patch.usedForTraining;

    if (Object.keys(values).length === 0) return;

    await this.db.updateTable('sticker_generation').set(values).where('id', '=', id).execute();
  }

  async getById(id: string) {
    return this.db
      .selectFrom('sticker_generation')
      .selectAll()
      .where('id', '=', id)
      .executeTakeFirst();
  }
}
