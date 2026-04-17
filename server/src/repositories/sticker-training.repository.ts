import { Injectable } from '@nestjs/common';
import { Insertable, Kysely, Selectable, Updateable, sql } from 'kysely';
import { InjectKysely } from 'nestjs-kysely';
import { DummyValue, GenerateSql } from 'src/decorators';
import { DB } from 'src/schema';
import {
  StickerTrainingCheckpointTable,
  StickerTrainingRunTable,
  StickerTrainingSampleTable,
} from 'src/schema/tables/sticker-training.table';

type StickerTrainingRun = Selectable<StickerTrainingRunTable>;
type StickerTrainingCheckpoint = Selectable<StickerTrainingCheckpointTable>;

@Injectable()
export class StickerTrainingRepository {
  constructor(@InjectKysely() private db: Kysely<DB>) {}

  @GenerateSql({ params: [DummyValue.UUID] })
  getRun(id: string): Promise<StickerTrainingRun | undefined> {
    return this.db.selectFrom('sticker_training_run').selectAll().where('id', '=', id).executeTakeFirst();
  }

  async createRun(input: Insertable<StickerTrainingRunTable>): Promise<StickerTrainingRun> {
    return this.db.insertInto('sticker_training_run').values(input).returningAll().executeTakeFirstOrThrow();
  }

  @GenerateSql({ params: [DummyValue.UUID] })
  async updateRun(id: string, input: Updateable<StickerTrainingRunTable>): Promise<void> {
    await this.db.updateTable('sticker_training_run').set(input).where('id', '=', id).execute();
  }

  @GenerateSql()
  async hasActiveRun(): Promise<boolean> {
    const row = await this.db
      .selectFrom('sticker_training_run')
      .select('id')
      .where('status', 'in', ['queued', 'running'])
      .executeTakeFirst();

    return !!row;
  }

  @GenerateSql({ params: [DummyValue.UUID] })
  async createSample(input: Insertable<StickerTrainingSampleTable>): Promise<void> {
    await this.db.insertInto('sticker_training_sample').values(input).execute();
  }

  @GenerateSql({ params: [DummyValue.DATE] })
  async countEligibleSince(lastAcceptedAt: Date | null): Promise<number> {
    const result = await this.db
      .selectFrom('sticker_training_sample as sts')
      .leftJoin('user_metadata as um', (join) =>
        join.onRef('um.userId', '=', 'sts.userId').on('um.key', '=', sql.lit('preferences')),
      )
      .select((eb) => eb.fn.countAll<number>().as('count'))
      .where((eb) => (lastAcceptedAt ? eb('sts.acceptedAt', '>', lastAcceptedAt) : eb.val(true)))
      .where(
        sql<boolean>`coalesce(("um"."value" -> 'stickerTraining' ->> 'useEditsForModelTraining')::boolean, false) = true`,
      )
      .executeTakeFirstOrThrow();

    return result.count;
  }

  @GenerateSql({ params: [DummyValue.NUMBER] })
  async countLatestEligible(limit: number): Promise<number> {
    const result = await this.db
      .selectFrom((qb) =>
        qb
          .selectFrom('sticker_training_sample as sts')
          .leftJoin('user_metadata as um', (join) =>
            join.onRef('um.userId', '=', 'sts.userId').on('um.key', '=', sql.lit('preferences')),
          )
          .select('sts.id')
          .where(
            sql<boolean>`coalesce(("um"."value" -> 'stickerTraining' ->> 'useEditsForModelTraining')::boolean, false) = true`,
          )
          .orderBy('sts.acceptedAt', 'desc')
          .limit(limit)
          .as('eligible_samples'),
      )
      .select((eb) => eb.fn.countAll<number>().as('count'))
      .executeTakeFirstOrThrow();

    return result.count;
  }

  @GenerateSql()
  async getLatestEligibleAcceptedAt(): Promise<Date | null> {
    const row = await this.db
      .selectFrom('sticker_training_sample as sts')
      .leftJoin('user_metadata as um', (join) =>
        join.onRef('um.userId', '=', 'sts.userId').on('um.key', '=', sql.lit('preferences')),
      )
      .select('sts.acceptedAt')
      .where(
        sql<boolean>`coalesce(("um"."value" -> 'stickerTraining' ->> 'useEditsForModelTraining')::boolean, false) = true`,
      )
      .orderBy('sts.acceptedAt', 'desc')
      .executeTakeFirst();

    return row?.acceptedAt ?? null;
  }

  @GenerateSql({ params: ['default'] })
  async getCheckpoint(key: string): Promise<StickerTrainingCheckpoint | undefined> {
    return this.db.selectFrom('sticker_training_checkpoint').selectAll().where('key', '=', key).executeTakeFirst();
  }

  @GenerateSql({ params: ['default'] })
  async upsertCheckpoint(key: string, input: Updateable<StickerTrainingCheckpointTable>) {
    await this.db
      .insertInto('sticker_training_checkpoint')
      .values({ key, ...input })
      .onConflict((oc) => oc.column('key').doUpdateSet(input))
      .execute();
  }
}
