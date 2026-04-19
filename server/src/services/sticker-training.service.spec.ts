import { JobName, JobStatus } from 'src/enum';
import { StickerTrainingService } from 'src/services/sticker-training.service';
import { newTestService, ServiceMocks } from 'test/utils';

describe(StickerTrainingService.name, () => {
  let sut: StickerTrainingService;
  let mocks: ServiceMocks;

  const enabledTrainingConfig = {
    enabled: true,
    retrainThreshold: 5000,
    sampleWindowSize: 5000,
    pythonExecutable: 'python3',
    trainingScriptPath: '/tmp/train.py',
    trainingConfigPath: '/tmp/train.yaml',
    trainingWorkingDirectory: '/tmp',
    resultJsonPath: '/tmp/sticker-training-result.json',
    qualityGate: {
      minDiceScore: 0.8,
      minIouScore: 0.7,
      maxRuntimeSeconds: 14_400,
    },
  };

  beforeEach(() => {
    ({ sut, mocks } = newTestService(StickerTrainingService));
  });

  it('should work', () => {
    expect(sut).toBeDefined();
  });

  describe('handleCheckThreshold', () => {
    it('should skip if sticker training config is disabled', async () => {
      await expect(sut.handleCheckThreshold()).resolves.toBe(JobStatus.Skipped);
    });

    it('should queue a retrain run at threshold', async () => {
      mocks.systemMetadata.get.mockResolvedValue({ machineLearning: { stickerTraining: enabledTrainingConfig } });
      mocks.stickerTraining.getCheckpoint.mockResolvedValue(undefined);
      mocks.stickerTraining.countEligibleSince.mockResolvedValue(5000);
      mocks.stickerTraining.hasActiveRun.mockResolvedValue(false);
      mocks.stickerTraining.countLatestEligible.mockResolvedValue(5000);
      mocks.stickerTraining.getLatestEligibleAcceptedAt.mockResolvedValue(new Date('2026-04-17T00:00:00Z'));
      mocks.stickerTraining.createRun.mockResolvedValue({ id: 'run-1' } as never);

      await expect(sut.handleCheckThreshold()).resolves.toBe(JobStatus.Success);
      expect(mocks.job.queue).toHaveBeenCalledWith({
        name: JobName.StickerTrainingRun,
        data: { runId: 'run-1', triggeredAt: expect.any(String) },
      });
    });

    it('should skip when there is an active run', async () => {
      mocks.systemMetadata.get.mockResolvedValue({ machineLearning: { stickerTraining: enabledTrainingConfig } });
      mocks.stickerTraining.getCheckpoint.mockResolvedValue(undefined);
      mocks.stickerTraining.countEligibleSince.mockResolvedValue(5000);
      mocks.stickerTraining.hasActiveRun.mockResolvedValue(true);

      await expect(sut.handleCheckThreshold()).resolves.toBe(JobStatus.Skipped);
      expect(mocks.job.queue).not.toHaveBeenCalledWith(expect.objectContaining({ name: JobName.StickerTrainingRun }));
    });
  });

  describe('handleEvaluate', () => {
    it('should reject run if quality gate fails', async () => {
      mocks.systemMetadata.get.mockResolvedValue({ machineLearning: { stickerTraining: enabledTrainingConfig } });
      mocks.stickerTraining.getRun.mockResolvedValue({
        id: 'run-1',
        metrics: { dice: 0.9, iou: 0.1, runtimeSeconds: 1 },
      } as never);

      await expect(sut.handleEvaluate({ runId: 'run-1' })).resolves.toBe(JobStatus.Skipped);
      expect(mocks.stickerTraining.updateRun).toHaveBeenCalledWith(
        'run-1',
        expect.objectContaining({ status: 'rejected' }),
      );
    });
  });
});
