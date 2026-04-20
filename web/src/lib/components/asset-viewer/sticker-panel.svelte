<script lang="ts">
  import { getAssetMediaUrl } from '$lib/utils';
  import { getDimensions } from '$lib/utils/asset-utils';
  import { handleError } from '$lib/utils/handle-error';
  import { AssetMediaSize, type AssetResponseDto } from '@immich/sdk';
  import { IconButton, LoadingSpinner } from '@immich/ui';
  import { mdiClose, mdiPencil } from '@mdi/js';
  import { t } from 'svelte-i18n';

  interface Props {
    asset: AssetResponseDto;
    onClose: () => void;
  }

  let { asset, onClose }: Props = $props();

  type Mode = 'draw' | 'loading' | 'result' | 'editing';
  let mode: Mode = $state('draw');

  // Mini-map
  let overlayCanvas: HTMLCanvasElement | undefined = $state();
  let isDrawing = $state(false);
  let startRel = { x: 0, y: 0 };
  let currentRect: { x: number; y: number; w: number; h: number } | null = $state(null);
  let pointPos: { x: number; y: number } | null = $state(null);

  // Result
  let stickerDataUrl: string | null = $state(null);
  let originalImageData: ImageData | undefined;

  // Edit
  let editCanvas: HTMLCanvasElement | undefined = $state();
  let brushSize = $state(20);
  let editTool: 'erase' | 'restore' = $state('erase');
  let isPainting = $state(false);

  const dimensions = $derived(asset.exifInfo ? getDimensions(asset.exifInfo) : { width: 1920, height: 1080 });
  const originalW = $derived(dimensions.width ?? 1920);
  const originalH = $derived(dimensions.height ?? 1080);

  const previewUrl = $derived(
    getAssetMediaUrl({ id: asset.id, size: AssetMediaSize.Preview, cacheKey: asset.thumbhash }),
  );
  const thumbnailUrl = $derived(
    getAssetMediaUrl({ id: asset.id, size: AssetMediaSize.Thumbnail, cacheKey: asset.thumbhash }),
  );

  const CANVAS_W = 440;
  const CANVAS_H = $derived(Math.round(CANVAS_W * originalH / originalW));

  const getRelativePos = (e: MouseEvent): { x: number; y: number } => {
    if (!overlayCanvas) return { x: 0, y: 0 };
    const rect = overlayCanvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
  };

  const drawOverlay = () => {
    if (!overlayCanvas) return;
    const ctx = overlayCanvas.getContext('2d')!;
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    if (currentRect && currentRect.w > 0.005) {
      ctx.strokeStyle = '#4ade80';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        currentRect.x * CANVAS_W,
        currentRect.y * CANVAS_H,
        currentRect.w * CANVAS_W,
        currentRect.h * CANVAS_H,
      );
    } else if (pointPos) {
      const px = pointPos.x * CANVAS_W;
      const py = pointPos.y * CANVAS_H;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#4ade80';
      ctx.fill();
    }
  };

  const onMouseDown = (e: MouseEvent) => {
    const pos = getRelativePos(e);
    isDrawing = true;
    startRel = pos;
    currentRect = null;
    pointPos = null;
  };

  const onMouseMove = (e: MouseEvent) => {
    if (!isDrawing) return;
    const pos = getRelativePos(e);
    const dx = pos.x - startRel.x;
    const dy = pos.y - startRel.y;
    if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01) {
      currentRect = {
        x: Math.min(startRel.x, pos.x),
        y: Math.min(startRel.y, pos.y),
        w: Math.abs(dx),
        h: Math.abs(dy),
      };
      pointPos = null;
    }
    drawOverlay();
  };

  const onMouseUp = (e: MouseEvent) => {
    if (!isDrawing) return;
    isDrawing = false;
    if (!currentRect || currentRect.w < 0.01) {
      pointPos = getRelativePos(e);
      currentRect = null;
    }
    drawOverlay();
  };

  const onMouseLeave = () => {
    if (isDrawing) {
      isDrawing = false;
      // discard incomplete drag; keep any previously committed point/rect
      currentRect = null;
      pointPos = null;
      drawOverlay();
    }
  };

  const hasInput = $derived(
    (currentRect !== null && currentRect.w > 0.01 && currentRect.h > 0.01) || pointPos !== null,
  );

  const submit = async () => {
    if (!hasInput) return;
    mode = 'loading';

    try {
      const body: Record<string, unknown> = {};
      if (currentRect) {
        body.bbox = [currentRect.x * originalW, currentRect.y * originalH, currentRect.w * originalW, currentRect.h * originalH];
      } else if (pointPos) {
        body.pointCoords = [[pointPos.x * originalW, pointPos.y * originalH]];
      }

      const response = await fetch(`/api/assets/${asset.id}/sticker`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error(`${response.status}: ${response.statusText}`);

      const { mask } = (await response.json()) as { mask: string };
      await compositeWithMask(mask, !!currentRect);
      mode = 'result';
    } catch (err) {
      handleError(err, 'Failed to generate sticker');
      mode = 'draw';
    }
  };

  const loadImage = (src: string): Promise<HTMLImageElement> =>
    new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });

  const compositeWithMask = async (maskBase64: string, invertMask: boolean) => {
    const [img, maskImg] = await Promise.all([
      loadImage(previewUrl),
      loadImage(`data:image/png;base64,${maskBase64}`),
    ]);

    const w = img.naturalWidth;
    const h = img.naturalHeight;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, w, h);

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = w;
    maskCanvas.height = h;
    const maskCtx = maskCanvas.getContext('2d')!;
    maskCtx.drawImage(maskImg, 0, 0, w, h);
    const maskData = maskCtx.getImageData(0, 0, w, h);

    for (let i = 0; i < imageData.data.length; i += 4) {
      imageData.data[i + 3] = invertMask ? 255 - maskData.data[i] : maskData.data[i];
    }

    ctx.putImageData(imageData, 0, 0);
    originalImageData = ctx.getImageData(0, 0, w, h);
    stickerDataUrl = canvas.toDataURL('image/png');
  };

  const dataUrlToBlob = (dataUrl: string): Blob => {
    const [header, data] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)![1];
    const bytes = atob(data);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    return new Blob([arr], { type: mime });
  };

  const copyToClipboard = async () => {
    if (!stickerDataUrl) return;
    try {
      const blob = dataUrlToBlob(stickerDataUrl);
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    } catch (err) {
      handleError(err, $t('errors.unable_to_copy_to_clipboard'));
    }
  };

  const saveToDevice = () => {
    if (!stickerDataUrl) return;
    const a = document.createElement('a');
    a.href = stickerDataUrl;
    a.download = `sticker-${asset.id}.png`;
    a.click();
  };

  const openEdit = () => {
    mode = 'editing';
  };

  $effect(() => {
    if (mode === 'editing' && editCanvas && stickerDataUrl) {
      const dataUrl = stickerDataUrl;
      loadImage(dataUrl).then((img) => {
        if (!editCanvas) return;
        editCanvas.width = img.naturalWidth;
        editCanvas.height = img.naturalHeight;
        const ctx = editCanvas.getContext('2d')!;
        ctx.clearRect(0, 0, editCanvas.width, editCanvas.height);
        ctx.drawImage(img, 0, 0);
      });
    }
  });

  const closeEdit = () => {
    if (editCanvas) {
      stickerDataUrl = editCanvas.toDataURL('image/png');
    }
    mode = 'result';
  };

  const onEditMouseDown = (e: MouseEvent) => {
    isPainting = true;
    paint(e);
  };
  const onEditMouseMove = (e: MouseEvent) => {
    if (isPainting) paint(e);
  };
  const onEditMouseUp = () => {
    isPainting = false;
  };

  const paint = (e: MouseEvent) => {
    if (!editCanvas) return;
    const rect = editCanvas.getBoundingClientRect();
    const scaleX = editCanvas.width / rect.width;
    const scaleY = editCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const r = (brushSize / 2) * scaleX;
    const ctx = editCanvas.getContext('2d')!;

    if (editTool === 'erase') {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else if (originalImageData) {
      const imageData = ctx.getImageData(0, 0, editCanvas.width, editCanvas.height);
      const rx = Math.ceil(r);
      const cx = Math.floor(x);
      const cy = Math.floor(y);
      for (let dy = -rx; dy <= rx; dy++) {
        for (let dx = -rx; dx <= rx; dx++) {
          if (dx * dx + dy * dy > r * r) continue;
          const px = cx + dx;
          const py = cy + dy;
          if (px < 0 || py < 0 || px >= editCanvas.width || py >= editCanvas.height) continue;
          const i = (py * editCanvas.width + px) * 4;
          imageData.data[i] = originalImageData.data[i];
          imageData.data[i + 1] = originalImageData.data[i + 1];
          imageData.data[i + 2] = originalImageData.data[i + 2];
          imageData.data[i + 3] = originalImageData.data[i + 3];
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }
  };

  const reset = () => {
    mode = 'draw';
    currentRect = null;
    pointPos = null;
    stickerDataUrl = null;
    originalImageData = undefined;
    if (overlayCanvas) {
      const ctx = overlayCanvas.getContext('2d')!;
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  };
</script>

<div class="flex flex-col h-full bg-immich-dark-gray text-white select-none">
  <!-- Header -->
  <div class="flex items-center justify-between px-4 py-3 border-b border-white/10">
    <span class="font-semibold text-base">
      {#if mode === 'draw' || mode === 'loading'}
        {$t('generate_sticker')}
      {:else if mode === 'result'}
        {$t('sticker_generated')}
      {:else}
        {$t('edit_sticker')}
      {/if}
    </span>
    <div class="flex items-center gap-1">
      {#if mode === 'result'}
        <IconButton
          title={$t('edit_sticker')}
          icon={mdiPencil}
          color="secondary"
          variant="ghost"
          shape="round"
          size="small"
          onclick={openEdit}
        />
      {:else if mode === 'editing'}
        <IconButton
          title={$t('done')}
          icon={mdiPencil}
          color="primary"
          variant="ghost"
          shape="round"
          size="small"
          onclick={closeEdit}
        />
      {/if}
      <IconButton
        title={$t('close')}
        icon={mdiClose}
        color="secondary"
        variant="ghost"
        shape="round"
        size="small"
        onclick={onClose}
      />
    </div>
  </div>

  <!-- Body -->
  <div class="flex-1 flex flex-col items-center justify-center gap-4 p-4 overflow-y-auto">

    {#if mode === 'draw'}
      <p class="text-sm text-gray-300 text-center">{$t('sticker_draw_hint')}</p>

      <!-- Mini-map -->
      <div class="relative w-full rounded-lg overflow-hidden">
        <img src={thumbnailUrl} alt="" class="w-full block pointer-events-none" draggable="false" />
        <canvas
          bind:this={overlayCanvas}
          width={CANVAS_W}
          height={CANVAS_H}
          class="absolute inset-0 w-full h-full cursor-crosshair"
          style="aspect-ratio: {CANVAS_W} / {CANVAS_H}"
          onmousedown={onMouseDown}
          onmousemove={onMouseMove}
          onmouseup={onMouseUp}
          onmouseleave={onMouseLeave}
        ></canvas>
      </div>

      <button
        class="w-full rounded-xl py-2 font-medium transition-colors {hasInput
          ? 'bg-immich-primary hover:bg-immich-primary/90 text-white cursor-pointer'
          : 'bg-white/10 text-white/40 cursor-not-allowed'}"
        onclick={submit}
        disabled={!hasInput}
      >
        {$t('generate_sticker')}
      </button>

    {:else if mode === 'loading'}
      <LoadingSpinner />
      <p class="text-sm text-gray-400">{$t('loading')}</p>

    {:else if mode === 'result' && stickerDataUrl}
      <!-- Sticker preview with checkerboard background -->
      <div class="w-full rounded-lg overflow-hidden checkerboard">
        <img src={stickerDataUrl} alt="sticker" class="w-full block" draggable="false" />
      </div>

      <div class="flex gap-2 w-full">
        <button
          class="flex-1 rounded-xl py-2 bg-white/10 hover:bg-white/20 text-sm font-medium transition-colors"
          onclick={copyToClipboard}
        >
          {$t('copy_to_clipboard')}
        </button>
        <button
          class="flex-1 rounded-xl py-2 bg-white/10 hover:bg-white/20 text-sm font-medium transition-colors"
          onclick={saveToDevice}
        >
          {$t('save_to_device')}
        </button>
      </div>

      <button
        class="w-full rounded-xl py-2 bg-white/5 hover:bg-white/10 text-sm text-gray-400 transition-colors"
        onclick={reset}
      >
        {$t('try_again')}
      </button>

    {:else if mode === 'editing'}
      <!-- Edit canvas -->
      <div class="relative w-full rounded-lg overflow-hidden checkerboard">
        <canvas
          bind:this={editCanvas}
          class="w-full block cursor-crosshair"
          style="image-rendering: pixelated;"
          onmousedown={onEditMouseDown}
          onmousemove={onEditMouseMove}
          onmouseup={onEditMouseUp}
          onmouseleave={onEditMouseUp}
        ></canvas>
      </div>

      <div class="w-full flex items-center gap-3">
        <span class="text-sm text-gray-300 shrink-0">{$t('brush_size')}</span>
        <input
          type="range"
          min="5"
          max="60"
          bind:value={brushSize}
          class="flex-1 accent-immich-primary"
        />
      </div>

      <div class="flex gap-2 w-full">
        <button
          class="flex-1 rounded-xl py-2 font-medium text-sm transition-colors {editTool === 'restore'
            ? 'bg-immich-primary text-white'
            : 'bg-white/10 hover:bg-white/20 text-white'}"
          onclick={() => (editTool = 'restore')}
        >
          {$t('restore')}
        </button>
        <button
          class="flex-1 rounded-xl py-2 font-medium text-sm transition-colors {editTool === 'erase'
            ? 'bg-immich-primary text-white'
            : 'bg-white/10 hover:bg-white/20 text-white'}"
          onclick={() => (editTool = 'erase')}
        >
          {$t('erase')}
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  .checkerboard {
    background-color: #444;
    background-image: linear-gradient(45deg, #666 25%, transparent 25%),
      linear-gradient(-45deg, #666 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #666 75%),
      linear-gradient(-45deg, transparent 75%, #666 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
  }
</style>
