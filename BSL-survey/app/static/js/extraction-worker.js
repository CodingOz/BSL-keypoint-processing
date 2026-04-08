
import {
  FilesetResolver,
  HandLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';


async function decodeVideoToFrames(videoFile) {
  // Use VideoDecoder API (available in Workers) to extract frames without DOM
  const frames = [];
  const frameTimestamps = [];

  // Create an object URL the video decoder can read
  const url = URL.createObjectURL(videoFile);

  const videoBlob = await fetch(url).then(r => r.blob());
  URL.revokeObjectURL(url);

  // Demux using mp4box.js (loaded as a module)
  // This gives us raw H.264/H.265 chunks we can feed to VideoDecoder
  const mp4boxFile = await demuxMp4(videoBlob);   // see helper below
  const { chunks, fps, rotation } = mp4boxFile;

  const decoder = new VideoDecoder({
    output(videoFrame) {
      frames.push(videoFrame.clone());
      frameTimestamps.push(videoFrame.timestamp);
      videoFrame.close();
    },
    error(e) { console.error('VideoDecoder error:', e); }
  });

  decoder.configure({
    codec: mp4boxFile.codec,   // e.g. 'avc1.42001e'
    codedWidth: mp4boxFile.width,
    codedHeight: mp4boxFile.height,
  });

  for (const chunk of chunks) {
    decoder.decode(chunk);
  }
  await decoder.flush();
  decoder.close();

  return { frames, frameTimestamps, fps, rotation };
}

async function extractV1(frames, frameTimestamps, fps, rotation) {
  // The legacy package is loaded as a script tag in the host page and accessed
  // via self.Hands. If using a bundler, import from 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js'
  
  const hands = new self.Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
  });

  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  const framesOut = [];
  const anomalousHands = [];

  // The legacy API is callback-based; wrap in a promise queue
  let resolveFrame;
  hands.onResults((results) => {
    if (resolveFrame) resolveFrame(results);
  });

  for (let i = 0; i < frames.length; i++) {
    const videoFrame = frames[i];

    // Draw the VideoFrame to an OffscreenCanvas to get an ImageBitmap
    const canvas = new OffscreenCanvas(videoFrame.codedWidth, videoFrame.codedHeight);
    const ctx = canvas.getContext('2d');

    // Apply rotation correction (mirrors your Python rotate_frame logic)
    applyRotation(ctx, rotation, videoFrame.codedWidth, videoFrame.codedHeight);
    ctx.drawImage(videoFrame, 0, 0);

    const imageBitmap = await canvas.transferToImageBitmap();

    // Send frame to the legacy MediaPipe pipeline
    const results = await new Promise((resolve) => {
      resolveFrame = resolve;
      hands.send({ image: imageBitmap });
    });

    imageBitmap.close();

    const handsData = { left: [], right: [] };

    if (results.multiHandLandmarks && results.multiHandedness) {
      const seenLabels = new Set();
      for (let h = 0; h < results.multiHandLandmarks.length; h++) {
        const handedness = results.multiHandedness[h];
        const label = handedness.label.toLowerCase(); // 'left' or 'right'
        const clusterId = label === 'left' ? 0 : 1;

        if (seenLabels.has(label)) {
          // Same hand detected twice — matches your handle_simultaneous_hands logic
          anomalousHands.push([i, label]);
          continue;
        }
        seenLabels.add(label);

        for (let lmIdx = 0; lmIdx < results.multiHandLandmarks[h].length; lmIdx++) {
          const lm = results.multiHandLandmarks[h][lmIdx];
          handsData[label].push({
            cluster_id: clusterId,
            landmark_id: lmIdx,
            x: lm.x,
            y: lm.y,
            z: lm.z,
          });
        }
      }
    }

    framesOut.push({
      frame_index: i,
      timestamp: fps > 0 ? i / fps : null,
      hands: handsData,
    });
  }

  hands.close();
  return { framesOut, anomalousHands };
}

async function extractV2(frames, frameTimestamps, fps, rotation) {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
  );

  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'CPU',  // Workers have no GPU access
    },
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    runningMode: 'VIDEO',  // VIDEO mode enables inter-frame tracking
  });

  const framesOut = [];
  const anomalousHands = [];

  for (let i = 0; i < frames.length; i++) {
    const videoFrame = frames[i];

    const canvas = new OffscreenCanvas(videoFrame.codedWidth, videoFrame.codedHeight);
    const ctx = canvas.getContext('2d');
    applyRotation(ctx, rotation, videoFrame.codedWidth, videoFrame.codedHeight);
    ctx.drawImage(videoFrame, 0, 0);

    // detectForVideo requires a monotonically increasing timestamp in ms
    const timestampMs = Math.round(frameTimestamps[i] / 1000);
    const imageBitmap = await canvas.transferToImageBitmap();

    // Tasks Vision API is synchronous in VIDEO mode
    const result = handLandmarker.detectForVideo(imageBitmap, timestampMs);
    imageBitmap.close();

    const handsData = { left: [], right: [] };

    if (result.landmarks && result.handedness) {
      const seenLabels = new Set();
      for (let h = 0; h < result.landmarks.length; h++) {
        // Tasks API returns handedness differently — check categoryName
        const handednessCategories = result.handedness[h];
        const label = handednessCategories[0].categoryName.toLowerCase();
        const clusterId = label === 'left' ? 0 : 1;

        if (seenLabels.has(label)) {
          anomalousHands.push([i, label]);
          continue;
        }
        seenLabels.add(label);

        for (let lmIdx = 0; lmIdx < result.landmarks[h].length; lmIdx++) {
          const lm = result.landmarks[h][lmIdx];
          handsData[label].push({
            cluster_id: clusterId,
            landmark_id: lmIdx,
            x: lm.x,
            y: lm.y,
            z: lm.z,
          });
        }
      }
    }

    framesOut.push({
      frame_index: i,
      timestamp: fps > 0 ? i / fps : null,
      hands: handsData,
    });
  }

  handLandmarker.close();
  return { framesOut, anomalousHands };
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'EXTRACT') return;
  const { slotId, videoFile, videoId, sign } = data;

  const progress = (stage, percent) =>
    self.postMessage({ type: 'PROGRESS', slotId, stage, percent });

  try {
    progress('decoding', 0);
    const { frames, frameTimestamps, fps, rotation } =
      await decodeVideoToFrames(videoFile, (p) => progress('decoding', p));

    const metadata = {
      video_id:        videoId,
      sign:            sign,
      fps:             fps,
      frame_count:     frames.length,
      rotation_applied: rotation,
      resolution:      [frames[0]?.codedWidth ?? 0, frames[0]?.codedHeight ?? 0],
    };

    progress('extracting_v1', 0);
    const v1 = await extractV1(
      frames, frameTimestamps, fps, rotation,
      (p) => progress('extracting_v1', p),
    );

    progress('extracting_v2', 0);
    const v2 = await extractV2(
      frames, frameTimestamps, fps, rotation,
      (p) => progress('extracting_v2', p),
    );

    // Release GPU/memory-backed VideoFrames immediately
    for (const frame of frames) frame.close();

    self.postMessage({
      type: 'DONE',
      slotId,
      payload: {
        v1: {
          metadata: { ...metadata, mediapipe_api: 'legacy_0.4.x' },
          frames:   v1.framesOut,
          anomalous_hands: v1.anomalousHands,
        },
        v2: {
          metadata: { ...metadata, mediapipe_api: 'tasks_vision_latest' },
          frames:   v2.framesOut,
          anomalous_hands: v2.anomalousHands,
        },
      },
    });

  } catch (err) {
    self.postMessage({ type: 'ERROR', slotId, message: err.message });
  }
};



function applyRotation(ctx, rotation, width, height) {
  if (rotation === 90) {
    ctx.translate(width, 0);
    ctx.rotate(Math.PI / 2);
  } else if (rotation === 180) {
    ctx.translate(width, height);
    ctx.rotate(Math.PI);
  } else if (rotation === 270) {
    ctx.translate(0, height);
    ctx.rotate(-Math.PI / 2);
  }
}