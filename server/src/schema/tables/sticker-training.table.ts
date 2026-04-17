import {
  Column,
  CreateDateColumn,
  ForeignKeyColumn,
  Generated,
  PrimaryColumn,
  PrimaryGeneratedColumn,
  Table,
  Timestamp,
  UpdateDateColumn,
} from '@immich/sql-tools';
import { UpdatedAtTrigger } from 'src/decorators';
import { AssetTable } from 'src/schema/tables/asset.table';
import { UserTable } from 'src/schema/tables/user.table';

@Table('sticker_training_sample')
export class StickerTrainingSampleTable {
  @PrimaryGeneratedColumn()
  id!: Generated<string>;

  @CreateDateColumn()
  createdAt!: Generated<Timestamp>;

  @Column({ type: 'timestamp with time zone' })
  acceptedAt!: Timestamp;

  @ForeignKeyColumn(() => UserTable, { onDelete: 'CASCADE', onUpdate: 'CASCADE' })
  userId!: string;

  @ForeignKeyColumn(() => AssetTable, { onDelete: 'SET NULL', onUpdate: 'CASCADE', nullable: true })
  sourceAssetId!: string | null;

  @ForeignKeyColumn(() => AssetTable, { onDelete: 'SET NULL', onUpdate: 'CASCADE', nullable: true })
  generatedMaskAssetId!: string | null;

  @ForeignKeyColumn(() => AssetTable, { onDelete: 'SET NULL', onUpdate: 'CASCADE', nullable: true })
  editedMaskAssetId!: string | null;

  @Column({ type: 'jsonb', nullable: true })
  prompt!: Record<string, unknown> | null;

  @Column({ type: 'jsonb', nullable: true })
  feedback!: Record<string, unknown> | null;
}

@Table('sticker_training_run')
@UpdatedAtTrigger('sticker_training_run_updated_at')
export class StickerTrainingRunTable {
  @PrimaryGeneratedColumn()
  id!: Generated<string>;

  @CreateDateColumn()
  createdAt!: Generated<Timestamp>;

  @UpdateDateColumn()
  updatedAt!: Generated<Timestamp>;

  @Column()
  status!: 'queued' | 'running' | 'failed' | 'completed' | 'rejected';

  @Column({ type: 'integer' })
  retrainThreshold!: number;

  @Column({ type: 'integer' })
  sampleWindowSize!: number;

  @Column({ type: 'integer' })
  sampleCount!: number;

  @Column({ type: 'timestamp with time zone', nullable: true })
  trainingDataMaxAcceptedAt!: Timestamp | null;

  @Column({ type: 'timestamp with time zone', nullable: true })
  startedAt!: Timestamp | null;

  @Column({ type: 'timestamp with time zone', nullable: true })
  finishedAt!: Timestamp | null;

  @Column({ type: 'integer', nullable: true })
  processExitCode!: number | null;

  @Column({ type: 'text', nullable: true })
  failureReason!: string | null;

  @Column({ type: 'jsonb', nullable: true })
  metrics!: Record<string, unknown> | null;

  @Column({ type: 'jsonb', nullable: true })
  qualityGate!: Record<string, unknown> | null;
}

@Table('sticker_training_checkpoint')
@UpdatedAtTrigger('sticker_training_checkpoint_updated_at')
export class StickerTrainingCheckpointTable {
  @PrimaryColumn({ type: 'character varying' })
  key!: string;

  @UpdateDateColumn()
  updatedAt!: Generated<Timestamp>;

  @Column({ type: 'timestamp with time zone', nullable: true })
  lastTriggeredAcceptedAt!: Timestamp | null;

  @Column({ type: 'timestamp with time zone', nullable: true })
  lastCompletedAcceptedAt!: Timestamp | null;

  @Column({ type: 'character varying', nullable: true })
  lastRunId!: string | null;
}
