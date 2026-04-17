import { Injectable } from '@nestjs/common';
import { OnEvent, OnJob } from 'src/decorators';
import { JobName, JobStatus, QueueName } from 'src/enum';
import { ArgOf } from 'src/repositories/event.repository';
import { BaseService } from 'src/services/base.service';
import { JobOf } from 'src/types';

const CHECKPOINT_KEY = 'default';

type TrainingResultPayload = {
  metrics?: { dice?: number; runtimeSeconds?: number; [key: string]: unknown };
  mlflow?: { runId?: string; [key: string]: unknown };
  [key: string]: unknown;
};

@Injectable()
export class StickerTrainingService extends BaseService {
  @OnEvent({ name: 'ConfigInit' })
  async onConfigInit({ newConfig }: ArgOf<'ConfigInit'>) {
    if (!newConfig.machineLearning.stickerTraining.enabled) {
      return;
    }

    await this.jobRepository.queue({ name: JobName.StickerTrainingCheckThreshold });
  }

  @OnEvent({ name: 'ConfigUpdate', server: true })
  async onConfigUpdate({ newConfig }: ArgOf<'ConfigUpdate'>) {
    if (!newConfig.machineLearning.stickerTraining.enabled) {
      return;
    }

    await this.jobRepository.queue({ name: JobName.StickerTrainingCheckThreshold });
  }

  @OnJob({ name: JobName.StickerTrainingCheckThreshold, queue: QueueName.StickerTraining })
  async handleCheckThreshold(): Promise<JobStatus> {
    const { machineLearning } = await this.getConfig({ withCache: false });
    const { stickerTraining } = machineLearning;
    if (!stickerTraining.enabled) {
      return JobStatus.Skipped;
    }

    const checkpoint = await this.stickerTrainingRepository.getCheckpoint(CHECKPOINT_KEY);
    const eligibleSinceLastRun = await this.stickerTrainingRepository.countEligibleSince(checkpoint?.lastCompletedAcceptedAt ?? null);
    if (eligibleSinceLastRun < stickerTraining.retrainThreshold) {
      return JobStatus.Skipped;
    }

    const activeRun = await this.stickerTrainingRepository.hasActiveRun();
    if (activeRun) {
      this.logger.log('Skipping sticker retrain trigger because another training run is active');
      return JobStatus.Skipped;
    }

    const latestEligibleAcceptedAt = await this.stickerTrainingRepository.getLatestEligibleAcceptedAt();
    const run = await this.stickerTrainingRepository.createRun({
      status: 'queued',
      retrainThreshold: stickerTraining.retrainThreshold,
      sampleWindowSize: stickerTraining.sampleWindowSize,
      sampleCount: await this.stickerTrainingRepository.countLatestEligible(stickerTraining.sampleWindowSize),
      trainingDataMaxAcceptedAt: latestEligibleAcceptedAt,
    });

    await this.stickerTrainingRepository.upsertCheckpoint(CHECKPOINT_KEY, {
      lastRunId: run.id,
      lastTriggeredAcceptedAt: latestEligibleAcceptedAt,
    });

    await this.jobRepository.queue({
      name: JobName.StickerTrainingRun,
      data: { runId: run.id, triggeredAt: new Date().toISOString() },
    });

    return JobStatus.Success;
  }

  @OnJob({ name: JobName.StickerTrainingRun, queue: QueueName.StickerTraining })
  async handleTrainingRun(data: JobOf<JobName.StickerTrainingRun>): Promise<JobStatus> {
    const { machineLearning } = await this.getConfig({ withCache: false });
    const { stickerTraining } = machineLearning;
    if (!stickerTraining.enabled) {
      return JobStatus.Skipped;
    }

    const run = await this.stickerTrainingRepository.getRun(data.runId);
    if (!run) {
      this.logger.warn(`Sticker training run not found: ${data.runId}`);
      return JobStatus.Failed;
    }

    await this.stickerTrainingRepository.updateRun(run.id, {
      status: 'running',
      startedAt: new Date(),
      failureReason: null,
    });

    const args = [
      stickerTraining.trainingScriptPath,
      '--sample-window-size',
      String(stickerTraining.sampleWindowSize),
      '--run-id',
      run.id,
      '--output-json',
      stickerTraining.resultJsonPath,
    ];

    const childProcess = this.processRepository.spawn(stickerTraining.pythonExecutable, args, {
      cwd: stickerTraining.trainingWorkingDirectory || undefined,
      env: globalThis.process.env,
    });

    const maxRuntimeMs = stickerTraining.qualityGate.maxRuntimeSeconds * 1000;
    const startedAt = Date.now();

    const exitCode = await new Promise<number>((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          childProcess.kill();
          reject(new Error('Sticker training process exceeded max runtime'));
        }
      }, maxRuntimeMs);

      childProcess.once('error', (error) => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          reject(error);
        }
      });

      childProcess.once('exit', (code) => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(code ?? 1);
        }
      });
    }).catch(async (error: Error) => {
      await this.stickerTrainingRepository.updateRun(run.id, {
        status: 'failed',
        finishedAt: new Date(),
        processExitCode: 1,
        failureReason: error.message,
        metrics: { runtimeSeconds: Math.floor((Date.now() - startedAt) / 1000) },
      });

      throw error;
    });

    if (exitCode !== 0) {
      await this.stickerTrainingRepository.updateRun(run.id, {
        status: 'failed',
        finishedAt: new Date(),
        processExitCode: exitCode,
        failureReason: `Training script exited with code ${exitCode}`,
        metrics: { runtimeSeconds: Math.floor((Date.now() - startedAt) / 1000) },
      });
      return JobStatus.Failed;
    }

    let parsedResult: TrainingResultPayload = {};
    try {
      const payload = await this.systemMetadataRepository.readFile(stickerTraining.resultJsonPath);
      parsedResult = JSON.parse(payload) as TrainingResultPayload;
    } catch (error) {
      await this.stickerTrainingRepository.updateRun(run.id, {
        status: 'failed',
        finishedAt: new Date(),
        processExitCode: exitCode,
        failureReason: 'Unable to parse training result output JSON',
      });
      return JobStatus.Failed;
    }

    const runtimeSeconds = Math.floor((Date.now() - startedAt) / 1000);
    await this.stickerTrainingRepository.updateRun(run.id, {
      status: 'completed',
      finishedAt: new Date(),
      processExitCode: 0,
      metrics: { ...parsedResult.metrics, runtimeSeconds },
    });

    await this.jobRepository.queue({
      name: JobName.StickerTrainingEvaluate,
      data: { runId: run.id },
    });

    return JobStatus.Success;
  }

  @OnJob({ name: JobName.StickerTrainingEvaluate, queue: QueueName.StickerTraining })
  async handleEvaluate(data: JobOf<JobName.StickerTrainingEvaluate>): Promise<JobStatus> {
    const { machineLearning } = await this.getConfig({ withCache: false });
    const { stickerTraining } = machineLearning;
    if (!stickerTraining.enabled) {
      return JobStatus.Skipped;
    }

    const run = await this.stickerTrainingRepository.getRun(data.runId);
    if (!run) {
      return JobStatus.Failed;
    }

    const metrics = (run.metrics ?? {}) as { dice?: number; runtimeSeconds?: number; [key: string]: unknown };
    const dice = typeof metrics.dice === 'number' ? metrics.dice : null;
    const runtimeSeconds = typeof metrics.runtimeSeconds === 'number' ? metrics.runtimeSeconds : null;

    const passDice = dice !== null && dice >= stickerTraining.qualityGate.minDiceScore;
    const passRuntime = runtimeSeconds !== null && runtimeSeconds <= stickerTraining.qualityGate.maxRuntimeSeconds;
    const passed = passDice && passRuntime;

    await this.stickerTrainingRepository.updateRun(run.id, {
      status: passed ? 'completed' : 'rejected',
      qualityGate: {
        passed,
        passDice,
        passRuntime,
        expectedMinDiceScore: stickerTraining.qualityGate.minDiceScore,
        expectedMaxRuntimeSeconds: stickerTraining.qualityGate.maxRuntimeSeconds,
      },
      failureReason: passed ? null : 'Quality gate failed',
    });

    if (passed && run.trainingDataMaxAcceptedAt) {
      await this.stickerTrainingRepository.upsertCheckpoint(CHECKPOINT_KEY, {
        lastRunId: run.id,
        lastCompletedAcceptedAt: run.trainingDataMaxAcceptedAt,
      });
    }

    return passed ? JobStatus.Success : JobStatus.Skipped;
  }
}
