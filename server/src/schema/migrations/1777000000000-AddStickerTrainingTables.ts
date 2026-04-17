import { Kysely, sql } from 'kysely';

export async function up(db: Kysely<any>): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS "sticker_training_sample" (
      "id" uuid PRIMARY KEY DEFAULT immich_uuid_v7(),
      "createdAt" timestamptz NOT NULL DEFAULT now(),
      "acceptedAt" timestamptz NOT NULL,
      "userId" uuid NOT NULL,
      "sourceAssetId" uuid,
      "generatedMaskAssetId" uuid,
      "editedMaskAssetId" uuid,
      "prompt" jsonb,
      "feedback" jsonb
    );
  `.execute(db);

  await sql`CREATE INDEX IF NOT EXISTS "IDX_sticker_training_sample_acceptedAt" ON "sticker_training_sample" ("acceptedAt");`.execute(
    db,
  );
  await sql`CREATE INDEX IF NOT EXISTS "IDX_sticker_training_sample_userId" ON "sticker_training_sample" ("userId");`.execute(
    db,
  );

  await sql`
    ALTER TABLE "sticker_training_sample"
      ADD CONSTRAINT "FK_sticker_training_sample_user"
      FOREIGN KEY ("userId") REFERENCES "user"("id")
      ON DELETE CASCADE ON UPDATE CASCADE;
  `.execute(db);

  await sql`
    ALTER TABLE "sticker_training_sample"
      ADD CONSTRAINT "FK_sticker_training_sample_source_asset"
      FOREIGN KEY ("sourceAssetId") REFERENCES "asset"("id")
      ON DELETE SET NULL ON UPDATE CASCADE;
  `.execute(db);

  await sql`
    ALTER TABLE "sticker_training_sample"
      ADD CONSTRAINT "FK_sticker_training_sample_generated_mask_asset"
      FOREIGN KEY ("generatedMaskAssetId") REFERENCES "asset"("id")
      ON DELETE SET NULL ON UPDATE CASCADE;
  `.execute(db);

  await sql`
    ALTER TABLE "sticker_training_sample"
      ADD CONSTRAINT "FK_sticker_training_sample_edited_mask_asset"
      FOREIGN KEY ("editedMaskAssetId") REFERENCES "asset"("id")
      ON DELETE SET NULL ON UPDATE CASCADE;
  `.execute(db);

  await sql`
    CREATE TABLE IF NOT EXISTS "sticker_training_run" (
      "id" uuid PRIMARY KEY DEFAULT immich_uuid_v7(),
      "createdAt" timestamptz NOT NULL DEFAULT now(),
      "updatedAt" timestamptz NOT NULL DEFAULT now(),
      "status" varchar NOT NULL,
      "retrainThreshold" integer NOT NULL,
      "sampleWindowSize" integer NOT NULL,
      "sampleCount" integer NOT NULL,
      "trainingDataMaxAcceptedAt" timestamptz,
      "startedAt" timestamptz,
      "finishedAt" timestamptz,
      "processExitCode" integer,
      "failureReason" text,
      "metrics" jsonb,
      "qualityGate" jsonb
    );
  `.execute(db);

  await sql`CREATE INDEX IF NOT EXISTS "IDX_sticker_training_run_createdAt" ON "sticker_training_run" ("createdAt");`.execute(
    db,
  );
  await sql`CREATE INDEX IF NOT EXISTS "IDX_sticker_training_run_status" ON "sticker_training_run" ("status");`.execute(
    db,
  );

  await sql`
    CREATE TABLE IF NOT EXISTS "sticker_training_checkpoint" (
      "key" varchar PRIMARY KEY,
      "updatedAt" timestamptz NOT NULL DEFAULT now(),
      "lastTriggeredAcceptedAt" timestamptz,
      "lastCompletedAcceptedAt" timestamptz,
      "lastRunId" varchar
    );
  `.execute(db);

  await sql`
    CREATE OR REPLACE FUNCTION sticker_training_run_updated_at() RETURNS trigger AS $$
      BEGIN
        NEW."updatedAt" = now();
        RETURN NEW;
      END
    $$ LANGUAGE plpgsql;
  `.execute(db);

  await sql`
    CREATE OR REPLACE TRIGGER "update_sticker_training_run_updatedAt"
    BEFORE UPDATE ON "sticker_training_run"
    FOR EACH ROW
    EXECUTE FUNCTION sticker_training_run_updated_at();
  `.execute(db);

  await sql`
    CREATE OR REPLACE FUNCTION sticker_training_checkpoint_updated_at() RETURNS trigger AS $$
      BEGIN
        NEW."updatedAt" = now();
        RETURN NEW;
      END
    $$ LANGUAGE plpgsql;
  `.execute(db);

  await sql`
    CREATE OR REPLACE TRIGGER "update_sticker_training_checkpoint_updatedAt"
    BEFORE UPDATE ON "sticker_training_checkpoint"
    FOR EACH ROW
    EXECUTE FUNCTION sticker_training_checkpoint_updated_at();
  `.execute(db);
}

export async function down(db: Kysely<any>): Promise<void> {
  await sql`DROP TRIGGER IF EXISTS "update_sticker_training_checkpoint_updatedAt" ON "sticker_training_checkpoint";`.execute(db);
  await sql`DROP FUNCTION IF EXISTS sticker_training_checkpoint_updated_at;`.execute(db);

  await sql`DROP TRIGGER IF EXISTS "update_sticker_training_run_updatedAt" ON "sticker_training_run";`.execute(db);
  await sql`DROP FUNCTION IF EXISTS sticker_training_run_updated_at;`.execute(db);

  await sql`DROP TABLE IF EXISTS "sticker_training_checkpoint";`.execute(db);
  await sql`DROP TABLE IF EXISTS "sticker_training_run";`.execute(db);
  await sql`DROP TABLE IF EXISTS "sticker_training_sample";`.execute(db);
}
