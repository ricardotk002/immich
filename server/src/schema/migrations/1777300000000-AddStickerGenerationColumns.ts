import { Kysely, sql } from 'kysely';

export async function up(db: Kysely<any>): Promise<void> {
  await sql`ALTER TABLE "sticker_generation" ADD COLUMN IF NOT EXISTS "s3StickerKey" text`.execute(db);
  await sql`ALTER TABLE "sticker_generation" ADD COLUMN IF NOT EXISTS "usedForTrainingAt" timestamp with time zone`.execute(db);
}

export async function down(db: Kysely<any>): Promise<void> {
  await sql`ALTER TABLE "sticker_generation" DROP COLUMN IF EXISTS "usedForTrainingAt"`.execute(db);
  await sql`ALTER TABLE "sticker_generation" DROP COLUMN IF EXISTS "s3StickerKey"`.execute(db);
}
