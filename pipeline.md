# Vilier Audio Pipeline

Tài liệu này mô tả luồng xử lý từ raw audio đến các artifact cuối cùng trong `outputs/<audio_id>/`. Trọng tâm là cách pipeline dùng diarization để phát hiện overlap, tách giọng trong vùng overlap, rồi đưa audio đã tách trở lại speaker track trước khi chạy ASR và labeling.

## 1. Raw Audio Input

Nguồn vào là một file audio hoặc một folder chứa nhiều file audio, được cấu hình qua:

```json
{
  "entrypoint": {
    "input_path": "inputs/real.wav",
    "output_path": "outputs",
    "sample_rate": 16000
  }
}
```

`run.sh` cũng nhận override qua biến môi trường:

```bash
INPUT_PATH=/path/to/audio.wav OUTPUT_PATH=outputs bash run.sh
```

Với mỗi file, pipeline tạo thư mục:

```text
outputs/<audio_id>/
```

Trong đó `<audio_id>` là tên file không có phần mở rộng.

## 2. Preprocess

Pipeline đọc raw audio bằng `soundfile`. Nếu `soundfile` không decode được, pipeline fallback sang `ffmpeg`.

Preprocess thực hiện:

- Convert về mono.
- Resample về `entrypoint.sample_rate`, mặc định `16000`.
- Chuẩn hoá dữ liệu thành `float32`.
- Ghi audio chuẩn hoá ra:

```text
outputs/<audio_id>/audio.standardized.wav
```

Từ bước này trở đi, timeline dùng sample rate chuẩn và timestamp tính theo audio đã chuẩn hoá.

## 3. VAD

VAD chạy trên `audio.standardized.wav` để tìm các đoạn có speech.

Output chính:

```text
outputs/<audio_id>/vad.json
outputs/<audio_id>/labels/vad.txt
outputs/<audio_id>/vad_audio/audio_1.wav
outputs/<audio_id>/vad_audio/audio_2.wav
...
```

Vai trò của VAD:

- Tạo speech intervals để review.
- Tạo `vad_audio/audio_*.wav` cho kiểm tra thủ công và hỗ trợ diarization chunking.
- Làm nền để tạo chunk cho một số backend diarization.

Lưu ý: `vad_audio/` không phải input ASR mặc định hiện tại. ASR mặc định chạy trên speaker-channel audio được tạo sau khi đã export speaker tracks.

## 4. Diarization Chunks

Pipeline tạo các chunk speech-only từ các đoạn VAD đã được sắp theo thứ tự thời gian. Bước này phục vụ các backend diarization cần input ngắn hơn, ví dụ Sortformer.

Output:

```text
outputs/<audio_id>/diarization_chunks/chunk_1.wav
outputs/<audio_id>/diarization_chunks/chunk_2.wav
...
```

### 4.1. Nguyên Tắc Ghép Chunk

Sau VAD, ta có danh sách utterance theo timeline gốc:

```text
vad_00000: start_1 -> end_1  -> vad_audio/audio_1.wav
vad_00001: start_2 -> end_2  -> vad_audio/audio_2.wav
...
vad_N:     start_N -> end_N  -> vad_audio/audio_N.wav
```

Pipeline tạo `chunk_1.wav` bằng cách lấy `vad_00000`, rồi thử nối thêm `vad_00001`, `vad_00002`, ... theo đúng thứ tự. Với chunk đang bắt đầu tại `start_1`, pipeline chọn `n` lớn nhất sao cho:

```text
end_n - start_1 <= diarization.max_chunk_seconds
```

Mặc định:

```text
diarization.max_chunk_seconds = 180.0
```

Nói cách khác, giới hạn 180 giây được tính trên timeline gốc từ start của utterance đầu tiên trong chunk đến end của utterance cuối cùng trong chunk. Không phải chỉ tính tổng duration của các file `vad_audio/audio_*.wav`.

Ví dụ với `max_chunk_seconds = 180`:

```text
audio_1: start=10,  end=30
audio_2: start=70,  end=90
audio_3: start=150, end=185
audio_4: start=200, end=230
```

Khi tạo chunk từ `audio_1`:

```text
audio_1 -> audio_3: end_3 - start_1 = 185 - 10 = 175 <= 180
audio_1 -> audio_4: end_4 - start_1 = 230 - 10 = 220 > 180
```

Vậy `chunk_1.wav` sẽ chứa:

```text
audio_1 + audio_2 + audio_3
```

Sau đó `chunk_2.wav` bắt đầu từ `audio_4`.

### 4.2. Audio Bên Trong Chunk

Chunk diarization giữ đúng khoảng cách thời gian giữa các VAD utterance trong cùng cửa sổ `start_1 -> end_n`. Pipeline cắt speech từ waveform gốc/chuẩn hoá theo từng VAD interval, và nếu giữa hai utterance có khoảng silence/non-speech thì chèn đoạn zero audio tương ứng vào chunk.

Ví dụ:

```text
audio_1: 10 -> 30
audio_2: 70 -> 90
```

Trong timeline gốc có 40 giây silence/non-speech từ `30 -> 70`, nên trong `chunk_1.wav`:

```text
audio_1 nằm ở chunk time 0  -> 20
silence nằm ở chunk time 20 -> 60
audio_2 nằm ở chunk time 60 -> 80
```

### 4.3. Mapping Từ Chunk Về Timeline Gốc

Vì chunk có thể được cắt từ một phần timeline gốc và vẫn cần biết từng mảnh speech đến từ VAD nào, pipeline ghi mapping cho từng mảnh speech:

```json
{
  "vad_id": "vad_00001",
  "chunk_start": 60.0,
  "chunk_end": 80.0,
  "source_start": 70.0,
  "source_end": 90.0
}
```

Ý nghĩa:

- Trên `chunk_1.wav`, đoạn này nằm từ giây `60.0` đến `80.0`.
- Trên audio gốc, đoạn này tương ứng từ giây `70.0` đến `90.0`.

Sau khi diarization chạy trên chunk, pipeline dùng mapping này để đưa segment về timeline gốc:

```text
source_time = source_start + (chunk_time - chunk_start)
```

Ví dụ model trả:

```text
chunk_1, SPEAKER_00: 65.0 -> 75.0
```

Mapping ở trên cho biết đoạn `60.0 -> 80.0` của chunk tương ứng `70.0 -> 90.0` trên audio gốc, nên segment sau remap là:

```text
SPEAKER_00: 75.0 -> 85.0
```

Mapping này được ghi vào `manifest.timeline.json` trong `diarization_chunks`.

Với backend chạy toàn bộ audio như `pyannote`, `pyannote/speaker-diarization-community-1`, `pyannote/speaker-diarization-3.1`, PixIT hoặc DiariZen, diarization có thể chạy trực tiếp trên `audio.standardized.wav`. Chunk vẫn là artifact hỗ trợ/tracking.

## 5. Diarization

Diarization nhận audio chuẩn hoá hoặc diarization chunks và tạo danh sách speaker segments:

```text
segment.id
segment.speaker
segment.start
segment.end
segment.duration
```

Ví dụ:

```json
{
  "id": "00012_SPEAKER_01",
  "speaker": "SPEAKER_01",
  "start": 35.42,
  "end": 39.18
}
```

Các backend hiện có:

- `pyannote` với `pyannote/speaker-diarization-community-1`.
- `pyannote` với `pyannote/speaker-diarization-3.1`.
- `sortformer`.
- `pixit` / `pyannote_pixit`.
- `diarizen`.

Sau diarization, pipeline gọi `annotate_overlaps()` để đánh dấu segment nào có overlap với speaker khác. Segment overlap sẽ có:

```text
is_overlap = true
overlap_group_id = overlap_00000
```

## 6. Speaker Linking

Pipeline ghi artifact:

```text
outputs/<audio_id>/speaker_linking.json
```

Mục đích là mô tả cách speaker id nội bộ hoặc speaker id theo chunk được nối thành speaker id cuối cùng như `SPEAKER_00`, `SPEAKER_01`.

Với backend chạy full-audio như pyannote hoặc DiariZen, strategy thường là:

```text
native_global
```

Nghĩa là model đã trả speaker id ở scope toàn bộ run.

Với backend chunked như Sortformer, pipeline có thể dùng ECAPA speaker embedding để nối local speaker trong từng chunk về global speaker id. Khi có embedding, artifact ghi:

- `links`: mapping từ local speaker sang global speaker.
- `embeddings`: vector embedding dùng để liên kết.
- `embedding_model`: model embedding.

## 7. Optional Music Separation

Nếu bật:

```json
{
  "music_separation": {
    "enabled": true,
    "backend": "demucs",
    "model": "htdemucs"
  }
}
```

Pipeline chạy Demucs trước overlap separation.

Output:

```text
outputs/<audio_id>/music_cleaned.wav
```

Từ đây:

- Nếu music separation bật và chạy thành công, overlap separation và speaker track export dùng `music_cleaned.wav`.
- Nếu music separation tắt hoặc fail, pipeline dùng lại `audio.standardized.wav`.

## 8. Overlap Separation

Overlap separation chạy sau diarization và optional music separation, nhưng trước khi export `tracks/SPEAKER_*.wav`.

Luồng tổng quát:

```text
diarization segments
  -> detect overlapping speaker pairs
  -> slice mixed overlap audio
  -> run separation model
  -> assign separated sources back to speakers
  -> match RMS
  -> write overlap review artifacts
  -> reconstruct enhanced segment audio
  -> pass segment_audio_overrides into speaker track export
```

Stage này chỉ xử lý các vùng mà diarization cho thấy có ít nhất hai speaker nói chồng nhau. Nó không chạy separation trên toàn bộ cuộc hội thoại.

Config hiện tại dùng ClearVoice MossFormer2:

```json
{
  "overlap_separation": {
    "enabled": true,
    "backend": "clearvoice",
    "model_name": "alibabasglab/MossFormer2_SS_16K",
    "device": "auto",
    "overlap_threshold_seconds": 0.2
  }
}
```

SpeechBrain và SepReformer vẫn là backend tuỳ chọn:

```json
{
  "overlap_separation": {
    "enabled": true,
    "backend": "speechbrain",
    "model_name": "speechbrain/sepformer-wsj02mix"
  }
}
```

```json
{
  "overlap_separation": {
    "enabled": true,
    "backend": "sepreformer",
    "sepreformer_path": "SepReformer",
    "model_name": "SepReformer_Base_WSJ0"
  }
}
```

Nếu `overlap_separation.enabled=false`, hoặc model separation load fail, stage sẽ `SKIP`. Khi đó:

- `overlap/` không có artifact mới.
- `segment_audio_overrides` rỗng.
- `tracks/SPEAKER_*.wav` được export từ waveform gốc/cleaned, không có vùng overlap đã tách.

### 8.1. Detect Overlap Pairs

Input của bước này là danh sách diarization segments sau khi đã được remap về timeline gốc:

```text
segment.id
segment.speaker
segment.start
segment.end
segment.is_overlap
segment.overlap_group_id
```

Pipeline sort segments theo:

```text
(start, end, speaker)
```

Sau đó duyệt từng cặp segment. Một overlap pair hợp lệ khi:

- Hai segment thuộc hai speaker khác nhau.
- Khoảng giao nhau giữa hai segment lớn hơn hoặc bằng `overlap_threshold_seconds`.

Điều kiện tính overlap:

```text
overlap_start = max(seg1.start, seg2.start)
overlap_end   = min(seg1.end, seg2.end)
overlap_dur   = overlap_end - overlap_start

valid if:
  seg1.speaker != seg2.speaker
  overlap_dur >= overlap_threshold_seconds
```

Ví dụ:

```text
SPEAKER_00: 10.0 -> 13.0
SPEAKER_01: 11.5 -> 14.0
overlap:    11.5 -> 13.0
```

Nếu `overlap_threshold_seconds=0.2`, vùng `11.5 -> 13.0` hợp lệ vì dài `1.5s`.

Với nhiều speaker chồng nhau cùng lúc, implementation hiện tại xử lý theo pair. Nghĩa là nếu có ba speaker cùng overlap, pipeline sẽ tạo nhiều overlap pairs hai-speaker thay vì một bài toán separation ba-speaker.

### 8.2. Slice Mixed Overlap Audio

Pipeline cắt đúng vùng overlap từ waveform đang dùng cho speaker export:

- `music_cleaned.wav` nếu music separation đã chạy.
- `audio.standardized.wav` nếu không có music separation.

Đoạn overlap này là mixed audio, còn chứa cả hai speaker:

```text
mixed_audio = waveform[overlap_start:overlap_end]
```

Ví dụ với sample rate `16000`:

```text
overlap_start = 11.5s -> start_idx = 184000
overlap_end   = 13.0s -> end_idx   = 208000

mixed_audio = waveform[184000:208000]
```

Pipeline chỉ gửi `mixed_audio` ngắn này vào separation model. Phần non-overlap của segment không được gửi vào model, vì phần đó đã thuộc về một speaker rõ ràng theo diarization.

### 8.3. Run Separation Model

Separator nhận mixed overlap audio và trả về hai source:

```text
src1
src2
```

Với `backend=speechbrain`, pipeline dùng:

```text
speechbrain.inference.separation.SepformerSeparation
model_name = speechbrain/sepformer-wsj02mix
```

SpeechBrain SepFormer chạy ở 8 kHz, nên wrapper làm:

```text
input sample rate 16 kHz
  -> resample mixed_audio về 8 kHz
  -> separate_batch()
  -> lấy source channel 0 và 1
  -> resample source về 16 kHz
```

Với `backend=sepreformer`, wrapper cũng đưa mixed overlap về 8 kHz, pad theo stride của model, chạy SepReformer, rồi resample output về sample rate pipeline.

Với `backend=clearvoice` hoặc alias `backend=mossformer2`, pipeline dùng:

```text
clearvoice.ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
model_name = alibabasglab/MossFormer2_SS_16K
```

ClearVoice MossFormer2 là model speech separation 16 kHz, nên wrapper làm:

```text
input sample rate của pipeline
  -> resample mixed_audio về 16 kHz nếu cần
  -> gọi ClearVoice numpy API với shape [1, length]
  -> nhận output shape [speaker, batch, length]
  -> lấy source speaker 0 và speaker 1
  -> resample source về sample rate pipeline nếu cần
```

Pipeline luôn chỉnh lại length của `src1` và `src2` để khớp chính xác số sample của overlap region.

Lý do phải match length:

- Resampling có thể lệch vài sample.
- Một số model có padding nội bộ.
- Khi reconstruct segment audio, phần thay thế phải cùng độ dài với vùng overlap gốc.

### 8.4. Assign Sources Back To Speakers

Sau khi model tách được hai source, pipeline phải quyết định source nào thuộc speaker nào.

Separation model chỉ trả:

```text
src1
src2
```

Nó không trả:

```text
src1 = SPEAKER_00
src2 = SPEAKER_01
```

Hiện tại logic gán nguồn dùng heuristic năng lượng và độ dài segment:

- Tính energy của `src1` và `src2`.
- Xác định segment nào dài hơn trong overlap pair.
- Xác định source nào lớn năng lượng hơn.
- Gán source tương ứng về `seg1` và `seg2`.

Pseudo-flow:

```text
energy1 = sum(src1 * src1)
energy2 = sum(src2 * src2)

seg1_is_longer = duration(seg1) >= duration(seg2)
src1_is_louder = energy1 >= energy2

if seg1_is_longer == src1_is_louder:
  seg1_audio = src1
  seg2_audio = src2
else:
  seg1_audio = src2
  seg2_audio = src1
```

Giới hạn hiện tại: bước này chưa dùng speaker embedding để identify source. Vì vậy trong một số overlap khó, source assignment có thể sai. Folder `overlap/` được ghi ra để nghe kiểm tra lại các trường hợp này.

### 8.5. RMS Matching

Sau khi gán source về speaker, pipeline scale volume của source đã tách để gần với mức RMS non-overlap của speaker đó.

Mục tiêu:

- Tránh source đã tách bị quá nhỏ so với phần còn lại của speaker track.
- Tránh source đã tách bị quá to gây clipping hoặc nghe không tự nhiên.
- Giữ transition giữa non-overlap và overlap bớt gắt.

Pipeline tìm RMS tham chiếu của speaker bằng cách lấy các phần của segment đó không nằm trong overlap pairs. Nếu không có phần non-overlap đủ tốt, fallback về RMS của mixed overlap nhân hệ số `0.7`.

Sau đó:

```text
source_rms = rms(separated_source)
target_rms = rms(non_overlap_audio_of_same_speaker)

scaled_source = separated_source * (target_rms / source_rms)
scaled_source = clip(scaled_source, -1.0, 1.0)
```

### 8.6. Write Overlap Artifacts

Với mỗi overlap pair, pipeline ghi các file review trong:

```text
outputs/<audio_id>/overlap/
```

Ví dụ:

```text
overlap/overlap_00001_mixed.wav
overlap/overlap_00001_SPEAKER_00.wav
overlap/overlap_00001_SPEAKER_01.wav
```

Ý nghĩa:

- `*_mixed.wav`: audio overlap trước separation.
- `*_SPEAKER_00.wav`: source sau separation được gán cho `SPEAKER_00`.
- `*_SPEAKER_01.wav`: source sau separation được gán cho `SPEAKER_01`.

Các path này cũng được ghi vào `manifest.timeline.json` dưới:

```text
overlap_separation.overlap_regions
```

Mỗi region record gồm:

```text
id
start
end
duration
speakers
segments
mixed_audio
separated_audio
```

Ví dụ:

```json
{
  "id": "overlap_00001",
  "start": 11.5,
  "end": 13.0,
  "duration": 1.5,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": ["00012_SPEAKER_00", "00013_SPEAKER_01"],
  "mixed_audio": "overlap/overlap_00001_mixed.wav",
  "separated_audio": {
    "SPEAKER_00": "overlap/overlap_00001_SPEAKER_00.wav",
    "SPEAKER_01": "overlap/overlap_00001_SPEAKER_01.wav"
  }
}
```

### 8.7. Reconstruct Enhanced Segment Audio

Đây là bước quan trọng sau khi đã tách âm từ overlap.

Pipeline không tạo một audio file mới cho toàn bộ cuộc hội thoại. Thay vào đó, nó tạo audio override cho từng diarization segment bị overlap:

```text
segment_audio[segment.id] = reconstructed_audio
```

Cách reconstruct:

1. Giữ nguyên các phần non-overlap của segment từ waveform gốc/cleaned.
2. Thay đúng phần overlap bằng source đã tách tương ứng với speaker của segment.
3. Nối các phần lại thành một audio segment hoàn chỉnh cùng duration với segment gốc.

Ví dụ:

```text
Segment SPEAKER_00: 10.0 -> 14.0

10.0 -> 11.5: giữ waveform gốc/cleaned
11.5 -> 13.0: thay bằng separated source của SPEAKER_00
13.0 -> 14.0: giữ waveform gốc/cleaned
```

Kết quả là một segment-level audio đã được cải thiện ở vùng overlap.

Nếu một segment có nhiều overlap regions, pipeline xử lý theo thứ tự thời gian:

```text
non-overlap part
overlap region 1 replaced by separated source
non-overlap gap
overlap region 2 replaced by separated source
remaining non-overlap part
```

### 8.8. Output Of This Phase

Function `apply_overlap_separation()` trả về:

```json
{
  "segment_audio": {
    "segment_id": "<reconstructed numpy audio>"
  },
  "overlap_regions": [
    {
      "id": "overlap_00001",
      "mixed_audio": "overlap/overlap_00001_mixed.wav",
      "separated_audio": {
        "SPEAKER_00": "overlap/overlap_00001_SPEAKER_00.wav",
        "SPEAKER_01": "overlap/overlap_00001_SPEAKER_01.wav"
      }
    }
  ]
}
```

Trong đó:

- `overlap_regions` là artifact/index để review và ghi manifest.
- `segment_audio` là output quan trọng để đưa vào speaker track export.

Nếu không phát hiện overlap pair hợp lệ:

```json
{
  "segment_audio": {},
  "overlap_regions": []
}
```

Trong log khi đó có thể thấy:

```text
regions=0
enhanced_segments=0
```

Điều này không nhất thiết là lỗi. Nó chỉ nghĩa là diarization không tạo cặp speaker overlap nào vượt `overlap_threshold_seconds`.

### 8.9. What Goes To The Next Phase

Stage tiếp theo là speaker track export. Pipeline truyền:

```text
segment_audio_overrides = overlap_result["segment_audio"]
```

Khi export từng segment:

- Nếu `segment.id` có trong `segment_audio_overrides`, dùng reconstructed audio đã thay overlap bằng separated source.
- Nếu không có override, slice audio bình thường từ waveform gốc/cleaned.

Vì vậy:

- `overlap/` là folder review/debug.
- `tracks/SPEAKER_*.wav` là nơi overlap separation thật sự được đưa vào audio speaker-level.
- `asr_audio/SPEAKER_*/*.wav` được sinh ra từ `tracks/`, nên ASR cũng nhận lợi ích từ overlap separation.

## 9. Speaker Track Export

Sau overlap separation, pipeline export full-duration track cho từng speaker:

```text
outputs/<audio_id>/tracks/SPEAKER_00.wav
outputs/<audio_id>/tracks/SPEAKER_01.wav
...
```

Đây là nơi output overlap separation được dùng tiếp.

Khi export track:

- Với segment không có overlap override, pipeline lấy audio trực tiếp từ waveform gốc/cleaned theo timestamp segment.
- Với segment có overlap override, pipeline dùng `segment_audio[segment.id]` đã reconstruct ở bước 8.6.
- Audio segment được cộng vào đúng vị trí timestamp trong full-duration speaker track.
- Track được clip về range `[-1.0, 1.0]`.

Nói ngắn gọn: sau khi tách overlap, audio đã tách không nằm riêng lẻ rồi dừng ở folder `overlap/`; nó được đưa trở lại speaker track qua `segment_audio_overrides`.

Folder `overlap/` là artifact để kiểm tra vùng overlap. Folder `tracks/` mới là audio speaker-level được pipeline dùng tiếp cho ASR.

## 10. Audacity Labels

Pipeline ghi label files:

```text
outputs/<audio_id>/labels/vad.txt
outputs/<audio_id>/labels/speakers.txt
outputs/<audio_id>/labels/SPEAKER_00.txt
outputs/<audio_id>/labels/SPEAKER_01.txt
...
```

Các file này dùng format:

```text
start_time<TAB>end_time<TAB>label
```

Có thể import trực tiếp vào Audacity để review timeline.

## 11. Speaker-Channel VAD For ASR

Sau khi có `tracks/SPEAKER_*.wav`, pipeline chạy VAD lần hai trên từng speaker track.

Output:

```text
outputs/<audio_id>/asr_audio/SPEAKER_00/audio_00001.wav
outputs/<audio_id>/asr_audio/SPEAKER_00/audio_00002.wav
outputs/<audio_id>/asr_audio/SPEAKER_01/audio_00001.wav
...
```

Mỗi item được ghi vào `manifest.timeline.json` trong `asr_segments`.

Đây là input ASR mặc định, vì nó đã đi qua:

```text
raw audio
-> preprocess
-> diarization
-> optional music separation
-> optional overlap separation
-> speaker track export
-> per-speaker VAD
```

## 12. ASR

Nếu bật:

```json
{
  "asr": {
    "enabled": true,
    "backend": "phowhisper_local",
    "model": "vinai/PhoWhisper-large",
    "language": "vi"
  }
}
```

Pipeline transcribe từng file trong `asr_audio/SPEAKER_*`.

Output:

```text
outputs/<audio_id>/transcript.json
```

Mỗi transcript record có:

```text
id
asr_segment_id
audio
start
end
duration
speaker
text
model
language
```

Vì ASR chạy trên speaker-channel audio, transcript đã có speaker id trực tiếp từ track tương ứng.

## 13. State Labeling

Nếu bật `state_labeling.enabled=true`, pipeline gửi transcript sang labeling runner, ví dụ Qwen, để gán nhãn như:

```text
complete
incomplete
```

Output:

```text
outputs/<audio_id>/state/index.json
outputs/<audio_id>/state/complete/*.wav
outputs/<audio_id>/state/complete/*.json
outputs/<audio_id>/state/incomplete/*.wav
outputs/<audio_id>/state/incomplete/*.json
```

Các record đã label cũng được ghi lại vào `transcript.json` và `manifest.timeline.json`.

## 14. Manifest And Resolved Config

Cuối pipeline, hệ thống ghi:

```text
outputs/<audio_id>/manifest.timeline.json
outputs/<audio_id>/config.resolved.json
```

`manifest.timeline.json` là index chính của run. Nó chứa:

- Source audio path.
- Standardized audio path.
- VAD segments.
- VAD review audio.
- Diarization chunks và source-time mapping.
- Speaker linking summary.
- Diarized speaker segments.
- Overlap regions và overlap audio artifacts.
- Speaker tracks.
- ASR segments.
- Transcript.
- State labeling summary.
- Audacity labels.

`config.resolved.json` ghi lại:

- Input/output đã resolve.
- Dry-run flag.
- Sample rate.
- Effective config.
- Backend/model/device thực tế của từng component.

File này dùng để biết run đang dùng model nào, ví dụ:

```text
diarization: pyannote/speaker-diarization-community-1
overlap_separation: alibabasglab/MossFormer2_SS_16K
asr: vinai/PhoWhisper-large
```

## 15. End-To-End Flow Summary

```text
raw audio
  -> audio.standardized.wav
  -> VAD
  -> vad.json, vad_audio/, labels/vad.txt
  -> diarization_chunks/
  -> diarization segments
  -> annotate overlaps
  -> speaker_linking.json
  -> optional Demucs music separation
  -> music_cleaned.wav
  -> optional overlap separation
  -> overlap/*.wav review artifacts
  -> segment_audio_overrides for overlapped diarization segments
  -> tracks/SPEAKER_*.wav
  -> per-speaker VAD
  -> asr_audio/SPEAKER_*/*.wav
  -> ASR
  -> transcript.json
  -> optional state labeling
  -> state/
  -> manifest.timeline.json
  -> config.resolved.json
```

## 16. What Happens After Overlap Separation

Sau khi tách được âm từ các đoạn overlap, pipeline làm ba việc:

1. Ghi artifact để review:

```text
overlap/overlap_00001_mixed.wav
overlap/overlap_00001_SPEAKER_00.wav
overlap/overlap_00001_SPEAKER_01.wav
```

2. Reconstruct lại audio cho từng segment bị overlap:

```text
segment_audio[segment.id]
```

3. Dùng reconstructed segment audio đó khi tạo full speaker tracks:

```text
tracks/SPEAKER_00.wav
tracks/SPEAKER_01.wav
```

Sau đó pipeline chạy per-speaker VAD trên `tracks/SPEAKER_*.wav`, rồi ASR trên `asr_audio/SPEAKER_*/*.wav`.

Vì vậy, folder `overlap/` chỉ là artifact kiểm tra. Tác dụng thật của separation nằm ở `tracks/SPEAKER_*.wav` và các file `asr_audio/` được sinh ra từ track đã cải thiện.
