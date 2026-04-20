import { Column, CreateDateColumn, ForeignKeyColumn, Generated, PrimaryGeneratedColumn, Table, Timestamp } from '@immich/sql-tools';
import { AssetTable } from 'src/schema/tables/asset.table';
import { UserTable } from 'src/schema/tables/user.table';

export type StickerQualityStatus = 'pending' | 'pass' | 'fail';

@Table('sticker_generation')
export class StickerGenerationTable {
  @PrimaryGeneratedColumn()
  id!: Generated<string>;

  @ForeignKeyColumn(() => UserTable, { onDelete: 'CASCADE', onUpdate: 'CASCADE' })
  userId!: string;

  @ForeignKeyColumn(() => AssetTable, { onDelete: 'CASCADE', onUpdate: 'CASCADE' })
  assetId!: string;

  @Column({ type: 'jsonb', nullable: true })
  bbox!: number[] | null;

  @Column({ type: 'jsonb', nullable: true })
  pointCoords!: number[][] | null;

  @Column({ nullable: true })
  mlSuggestedMask!: string | null;

  @Column({ type: 'bytea', nullable: true })
  mlSuggestedMaskData!: Buffer | null;

  @Column({ nullable: true })
  userSavedMask!: string | null;

  @Column({ type: 'integer', default: 0 })
  processingTimeMs!: Generated<number>;

  @Column({ type: 'integer', default: 1 })
  numTries!: Generated<number>;

  @Column({ type: 'integer', default: 0 })
  editedPixels!: Generated<number>;

  @Column({ type: 'boolean', default: false })
  saved!: Generated<boolean>;

  @Column({ type: 'boolean', default: false })
  usedForTraining!: Generated<boolean>;

  @Column({ default: 'pending' })
  qualityStatus!: Generated<StickerQualityStatus>;

  @Column({ type: 'timestamp with time zone', nullable: true })
  qualityCheckedAt!: Timestamp | null;

  @Column({ type: 'integer', nullable: true })
  qualityCheckVersion!: number | null;

  @Column({ type: 'text', nullable: true })
  qualityFailReasonsJson!: string | null;

  @Column({ type: 'text', nullable: true })
  retrainRunId!: string | null;

  @CreateDateColumn()
  createdAt!: Generated<Timestamp>;
}
