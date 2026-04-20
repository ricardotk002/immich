import { Kysely, sql } from 'kysely';

export async function up(db: Kysely<any>): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS "sticker_generation" (
      "id"                     uuid PRIMARY KEY DEFAULT immich_uuid_v7(),
      "userId"                 uuid NOT NULL REFERENCES "user"("id") ON DELETE CASCADE ON UPDATE CASCADE,
      "assetId"                uuid NOT NULL REFERENCES "asset"("id") ON DELETE CASCADE ON UPDATE CASCADE,
      "bbox"                   jsonb,
      "pointCoords"            jsonb,
      "mlSuggestedMask"        text,
      "mlSuggestedMaskData"    bytea,
      "userSavedMask"          text,
      "processingTimeMs"       integer NOT NULL DEFAULT 0,
      "numTries"               integer NOT NULL DEFAULT 1,
      "editedPixels"           integer NOT NULL DEFAULT 0,
      "saved"                  boolean NOT NULL DEFAULT false,
      "usedForTraining"        boolean NOT NULL DEFAULT false,
      "qualityStatus"          text NOT NULL DEFAULT 'pending',
      "qualityCheckedAt"       timestamp with time zone,
      "qualityCheckVersion"    integer,
      "qualityFailReasonsJson" text,
      "retrainRunId"           text,
      "createdAt"              timestamp with time zone NOT NULL DEFAULT now()
    );
  `.execute(db);

  await sql`CREATE INDEX IF NOT EXISTS "sticker_generation_userId_idx" ON "sticker_generation" ("userId");`.execute(db);
  await sql`CREATE INDEX IF NOT EXISTS "sticker_generation_assetId_idx" ON "sticker_generation" ("assetId");`.execute(db);
  await sql`CREATE INDEX IF NOT EXISTS "sticker_generation_qualityStatus_idx" ON "sticker_generation" ("qualityStatus");`.execute(db);
  await sql`CREATE INDEX IF NOT EXISTS "sticker_generation_training_idx" ON "sticker_generation" ("saved", "usedForTraining");`.execute(db);
}

export async function down(db: Kysely<any>): Promise<void> {
  await sql`DROP TABLE IF EXISTS "sticker_generation";`.execute(db);
}
