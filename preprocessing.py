import os
import pickle
from collections import Counter

import numpy as np
from mmsdk import mmdatasdk
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split


class CMU_MOSEI_Preprocessor:
    """
    Preprocessor untuk CMU-MOSEI dataset dengan timestep normalization
    dan handling class imbalance.

    Compatible dengan mmdatasdk untuk loading .csd files.
    """

    # ========================================================================
    # MOSEI Label Mapping untuk Ekman-6 Emotions
    # ========================================================================
    # MOSEI Labels format: [sentiment, happiness, sadness, anger, surprise, disgust, fear]
    # Index 0: sentiment (-3 to 3) - SKIP THIS!
    # Index 1-6: Ekman emotions (0 to 3)

    MOSEI_LABEL_INDICES = {
        "sentiment": 0,  # -3 to 3 (TIDAK DIPAKAI untuk Ekman-6)
        "happiness": 1,  # 0 to 3
        "sadness": 2,  # 0 to 3
        "anger": 3,  # 0 to 3
        "surprise": 4,  # 0 to 3
        "disgust": 5,  # 0 to 3
        "fear": 6,  # 0 to 3
    }

    # Mapping ke Ekman-6 standard order
    EKMAN_TO_MOSEI_INDEX = {
        0: 3,  # anger
        1: 5,  # disgust
        2: 6,  # fear
        3: 1,  # happiness
        4: 2,  # sadness
        5: 4,  # surprise
    }

    EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

    def __init__(self, data_dir, target_timestep=20, threshold=1.0, strategy="max"):
        """
        Args:
            data_dir: Directory containing .csd files
            target_timestep: Target timestep untuk normalisasi (default: 20)
            threshold: Threshold untuk emotion classification
            strategy: 'max', 'threshold', atau 'strict'
        """
        self.data_dir = data_dir
        self.target_timestep = target_timestep
        self.threshold = threshold
        self.strategy = strategy

        # Paths ke .csd files
        self.covarep_path = os.path.join(data_dir, "CMU_MOSEI_COVAREP.csd")
        self.labels_path = os.path.join(data_dir, "CMU_MOSEI_Labels.csd")
        self.text_path = os.path.join(data_dir, "CMU_MOSEI_TimestampedWordVectors.csd")
        self.visual_path = os.path.join(data_dir, "CMU_MOSEI_VisualOpenFace2.csd")

        print(f"\n{'='*70}")
        print("CMU-MOSEI Preprocessor Initialized")
        print(f"{'='*70}")
        print(f"Data directory: {data_dir}")
        print(f"Target timestep: {target_timestep}")
        print(f"Strategy: {strategy}")
        print(f"Threshold: {threshold}")
        print(f"{'='*70}\n")

    def verify_files(self):
        """Verify bahwa semua file .csd ada"""
        print("🔍 Verifying .csd files...")
        missing_files = []

        for path in [
            self.covarep_path,
            self.labels_path,
            self.text_path,
            self.visual_path,
        ]:
            if not os.path.exists(path):
                print(f"✗ ERROR: File not found: {path}")
                missing_files.append(path)
            else:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✓ Found: {os.path.basename(path)} ({file_size_mb:.2f} MB)")

        if missing_files:
            raise FileNotFoundError(f"Missing {len(missing_files)} file(s)")

        print("✓ All files verified!\n")
        return True

    def load_csd_files(self):
        """Load semua .csd files menggunakan mmdatasdk"""
        print("📂 Loading .csd files with mmdatasdk...")

        try:
            self.covarep_data = mmdatasdk.computational_sequence(self.covarep_path)
            print(f"✓ Loaded COVAREP (Audio)")

            self.labels_data = mmdatasdk.computational_sequence(self.labels_path)
            print(f"✓ Loaded Labels")

            self.text_data = mmdatasdk.computational_sequence(self.text_path)
            print(f"✓ Loaded Text (Word Vectors)")

            self.visual_data = mmdatasdk.computational_sequence(self.visual_path)
            print(f"✓ Loaded Visual (OpenFace2)")

            # Extract dictionaries
            self.covarep_dict = self.covarep_data.data
            self.labels_dict = self.labels_data.data
            self.text_dict = self.text_data.data
            self.visual_dict = self.visual_data.data

            print(f"\n📊 Dataset Statistics:")
            print(f"  - Labels: {len(self.labels_dict)} segments")
            print(f"  - Audio: {len(self.covarep_dict)} segments")
            print(f"  - Text: {len(self.text_dict)} segments")
            print(f"  - Visual: {len(self.visual_dict)} segments")

            # Verifikasi jumlah data
            if len(self.labels_dict) < 20000:
                print(
                    f"\n⚠️  WARNING: Expected ~23,453 segments, got {len(self.labels_dict)}"
                )
                print("   Dataset mungkin tidak complete atau corrupted!")
            else:
                print(f"\n✓ Dataset size looks correct (~23,453 expected)")

            return True

        except Exception as e:
            print(f"✗ Error loading files: {e}")
            import traceback

            traceback.print_exc()
            return False

    def extract_features_from_dataset(self, dataset_obj):
        """
        Extract numpy array dari mmdatasdk Dataset object

        Args:
            dataset_obj: Object dari mmdatasdk

        Returns:
            numpy array atau None jika gagal
        """
        if dataset_obj is None:
            return None

        try:
            # Method 1: Direct numpy conversion
            if isinstance(dataset_obj, np.ndarray):
                return dataset_obj

            # Method 2: Check if it has 'features' attribute
            if hasattr(dataset_obj, "features"):
                features = dataset_obj.features
                if isinstance(features, np.ndarray):
                    return features
                return np.array(features)

            # Method 3: Try direct conversion
            return np.array(dataset_obj)

        except Exception as e:
            return None

    def align_features(self, features):
        """
        Align features ke target_timestep menggunakan interpolasi

        Args:
            features: array dengan shape (original_timestep, feature_dim)

        Returns:
            aligned_features: array dengan shape (target_timestep, feature_dim)
        """
        if features is None or len(features) == 0:
            return None

        original_timestep = len(features)

        if original_timestep == self.target_timestep:
            return features

        # Jika feature 1D, ubah ke 2D
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        feature_dim = features.shape[1]
        aligned = np.zeros((self.target_timestep, feature_dim))

        # Interpolasi untuk setiap dimensi fitur
        x_original = np.linspace(0, 1, original_timestep)
        x_target = np.linspace(0, 1, self.target_timestep)

        for i in range(feature_dim):
            f = interp1d(
                x_original, features[:, i], kind="linear", fill_value="extrapolate"
            )
            aligned[:, i] = f(x_target)

        return aligned

    def convert_to_single_label(self, emotion_scores):
        """
        Convert multi-label emotion scores ke single label classification

        Args:
            emotion_scores: array [anger, disgust, fear, happiness, sadness, surprise]
                           values from 0 to 3

        Returns:
            label: integer 0-5 untuk Ekman-6, atau -1 jika tidak ada emotion
        """
        if self.strategy == "max":
            # Ambil emotion dengan score tertinggi
            max_idx = np.argmax(emotion_scores)
            max_score = emotion_scores[max_idx]

            # Jika score tertinggi masih terlalu rendah, return -1
            if max_score < self.threshold:
                return -1

            return max_idx

        elif self.strategy == "threshold":
            # Ambil semua emotion > threshold
            above_threshold = np.where(emotion_scores >= self.threshold)[0]

            if len(above_threshold) == 0:
                return -1
            elif len(above_threshold) == 1:
                return above_threshold[0]
            else:
                # Multiple emotions, ambil yang tertinggi
                max_idx = above_threshold[np.argmax(emotion_scores[above_threshold])]
                return max_idx

        elif self.strategy == "strict":
            # Hanya ambil jika HANYA satu emotion > threshold
            above_threshold = np.where(emotion_scores >= self.threshold)[0]

            if len(above_threshold) == 1:
                return above_threshold[0]
            else:
                return -1

        return -1

    def load_and_preprocess(self):
        """Load dan preprocess semua data"""

        # Verify files
        self.verify_files()

        # Load files
        if not self.load_csd_files():
            raise RuntimeError("Failed to load .csd files")

        # Process data
        print(f"\n{'='*70}")
        print("STEP 1: Processing Segments")
        print(f"{'='*70}\n")

        aligned_audio = []
        aligned_visual = []
        aligned_text = []
        labels_list = []
        emotion_scores_list = []
        video_ids = []
        segment_ids = []

        # Statistics
        stats = {
            "total": len(self.labels_dict),
            "missing_modality": 0,
            "invalid_labels": 0,
            "weak_emotion": 0,
            "extraction_error": 0,
            "success": 0,
        }

        error_samples = []

        print("🔄 Processing segments...")
        for idx, segment_id in enumerate(self.labels_dict.keys()):
            try:
                # Progress indicator
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(self.labels_dict)} segments...")

                # Check if all modalities exist
                if (
                    segment_id not in self.covarep_dict
                    or segment_id not in self.visual_dict
                    or segment_id not in self.text_dict
                ):
                    stats["missing_modality"] += 1
                    continue

                # ================================================================
                # Extract LABELS
                # ================================================================
                segment_labels_obj = self.labels_dict[segment_id]["features"]
                segment_labels = self.extract_features_from_dataset(segment_labels_obj)

                if segment_labels is None or len(segment_labels) == 0:
                    stats["invalid_labels"] += 1
                    if len(error_samples) < 3:
                        error_samples.append(
                            {
                                "id": segment_id,
                                "reason": "Could not extract labels",
                                "type": type(segment_labels_obj),
                            }
                        )
                    continue

                # Flatten jika perlu
                if len(segment_labels.shape) > 1:
                    segment_labels = segment_labels.flatten()

                # Verifikasi punya minimal 7 values (sentiment + 6 emotions)
                if len(segment_labels) < 7:
                    stats["invalid_labels"] += 1
                    if len(error_samples) < 3:
                        error_samples.append(
                            {
                                "id": segment_id,
                                "reason": f"Labels too short: {len(segment_labels)}",
                                "labels": segment_labels,
                            }
                        )
                    continue

                # Extract 6 Ekman emotions (SKIP index 0 = sentiment!)
                ekman_scores = np.zeros(6, dtype=np.float32)
                for ekman_idx, mosei_idx in self.EKMAN_TO_MOSEI_INDEX.items():
                    ekman_scores[ekman_idx] = segment_labels[mosei_idx]

                # Convert ke single label
                emotion_label = self.convert_to_single_label(ekman_scores)

                # Skip jika no emotion
                if emotion_label == -1:
                    stats["weak_emotion"] += 1
                    continue

                # ================================================================
                # Extract FEATURES dari semua modalities
                # ================================================================
                audio_obj = self.covarep_dict[segment_id]["features"]
                visual_obj = self.visual_dict[segment_id]["features"]
                text_obj = self.text_dict[segment_id]["features"]

                audio_features = self.extract_features_from_dataset(audio_obj)
                visual_features = self.extract_features_from_dataset(visual_obj)
                text_features = self.extract_features_from_dataset(text_obj)

                if (
                    audio_features is None
                    or visual_features is None
                    or text_features is None
                ):
                    stats["extraction_error"] += 1
                    if len(error_samples) < 5:
                        error_samples.append(
                            {
                                "id": segment_id,
                                "reason": "Could not extract features",
                                "audio": audio_features is not None,
                                "visual": visual_features is not None,
                                "text": text_features is not None,
                            }
                        )
                    continue

                # Align ke target_timestep
                audio_aligned = self.align_features(audio_features)
                visual_aligned = self.align_features(visual_features)
                text_aligned = self.align_features(text_features)

                if (
                    audio_aligned is None
                    or visual_aligned is None
                    or text_aligned is None
                ):
                    stats["extraction_error"] += 1
                    continue

                # Simpan
                aligned_audio.append(audio_aligned)
                aligned_visual.append(visual_aligned)
                aligned_text.append(text_aligned)
                labels_list.append(emotion_label)
                emotion_scores_list.append(ekman_scores)
                video_ids.append(segment_id.split("[")[0])
                segment_ids.append(segment_id)

                stats["success"] += 1

            except Exception as e:
                if len(error_samples) < 5:
                    import traceback

                    error_samples.append(
                        {
                            "id": segment_id,
                            "reason": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue

        # Print statistics
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total segments: {stats['total']}")
        print(f"✓ Successfully processed: {stats['success']}")
        print(f"✗ Skipped (missing modality): {stats['missing_modality']}")
        print(f"✗ Skipped (invalid labels): {stats['invalid_labels']}")
        print(f"✗ Skipped (weak/no emotion): {stats['weak_emotion']}")
        print(f"✗ Skipped (extraction error): {stats['extraction_error']}")

        if error_samples:
            print(f"\n⚠️  Sample errors (first {len(error_samples)}):")
            for err in error_samples:
                print(f"\n  ID: {err['id']}")
                print(f"  Reason: {err['reason']}")

        if stats["success"] == 0:
            raise RuntimeError("No samples processed! Check data format.")

        # Convert to numpy arrays
        processed_samples = {
            "audio": np.array(aligned_audio, dtype=np.float32),
            "visual": np.array(aligned_visual, dtype=np.float32),
            "text": np.array(aligned_text, dtype=np.float32),
            "labels": np.array(labels_list, dtype=np.int32),
            "emotion_scores": np.array(emotion_scores_list, dtype=np.float32),
            "video_ids": np.array(video_ids),
            "segment_ids": np.array(segment_ids),
        }

        # Print shapes
        print(f"\n{'='*70}")
        print("DATA SHAPES")
        print(f"{'='*70}")
        print(f"Audio:          {processed_samples['audio'].shape}")
        print(f"Visual:         {processed_samples['visual'].shape}")
        print(f"Text:           {processed_samples['text'].shape}")
        print(f"Labels:         {processed_samples['labels'].shape}")
        print(f"Emotion Scores: {processed_samples['emotion_scores'].shape}")

        # Label distribution
        print(f"\n{'='*70}")
        print("LABEL DISTRIBUTION (Before Balancing)")
        print(f"{'='*70}")
        label_counts = Counter(processed_samples["labels"])
        for i in range(6):
            count = label_counts.get(i, 0)
            pct = (
                (count / len(processed_samples["labels"])) * 100
                if len(processed_samples["labels"]) > 0
                else 0
            )
            print(
                f"Class {i} ({self.EMOTION_NAMES[i]:12s}): {count:5d} samples ({pct:5.2f}%)"
            )

        return processed_samples

    def handle_class_imbalance(
        self,
        samples,
        strategy="oversample",
        min_samples_per_class=200,
        max_oversample_ratio=5.0,  # NEW: Maximum allowed oversampling ratio
        target_mode="median",  # NEW: 'median', 'mean', or 'max'
        random_state=42,
    ):
        """
        Handle class imbalance - SAFE VERSION with oversampling limits

        Args:
            samples: Dictionary with processed samples
            strategy: 'oversample', 'undersample', atau None
            min_samples_per_class: Minimum samples untuk oversampling
            max_oversample_ratio: Maximum ratio of synthetic/original samples (default: 5.0)
            target_mode: How to determine target samples ('median', 'mean', 'max')
            random_state: Random seed untuk reproducibility
        """
        print(f"\n{'='*70}")
        print("STEP 2: Handling Class Imbalance (Safe Mode with Limits)")
        print(f"{'='*70}")
        print(f"Strategy: {strategy}")
        print(f"Min samples per class: {min_samples_per_class}")
        print(f"Max oversample ratio: {max_oversample_ratio}x (safety limit)")
        print(f"Target mode: {target_mode}")
        print(f"Random state: {random_state}")

        labels = samples["labels"]
        label_counts = Counter(labels)
        n_original_samples = len(labels)

        print(f"\nOriginal distribution:")
        print(f"Total samples: {n_original_samples}")
        for i in range(6):
            count = label_counts.get(i, 0)
            pct = (count / n_original_samples) * 100 if n_original_samples > 0 else 0
            print(
                f"  Class {i} ({self.EMOTION_NAMES[i]:12s}): {count:5d} samples ({pct:5.2f}%)"
            )

        # Verify all classes are present
        missing_classes = [i for i in range(6) if i not in label_counts]
        if missing_classes:
            print(f"\n⚠️  WARNING: Missing classes: {missing_classes}")
            print("   These classes have 0 samples and cannot be balanced!")

        if strategy == "oversample":
            from collections import defaultdict

            # Set random seed for reproducibility
            np.random.seed(random_state)

            # Group samples by class
            indices_by_class = defaultdict(list)
            for idx, label in enumerate(labels):
                indices_by_class[label].append(idx)

            # ================================================================
            # SAFE TARGET CALCULATION with Safety Limits
            # ================================================================
            class_counts = [
                label_counts.get(i, 0) for i in range(6) if i in label_counts
            ]

            if target_mode == "median":
                base_target = int(np.median(class_counts))
            elif target_mode == "mean":
                base_target = int(np.mean(class_counts))
            else:  # max
                base_target = max(class_counts)

            # Apply minimum threshold
            target_samples = max(min_samples_per_class, base_target)

            print(f"\n🎯 Target Calculation:")
            print(f"  Base target ({target_mode}): {base_target}")
            print(f"  After min threshold: {target_samples}")

            # ================================================================
            # SAFETY CHECK: Detect extreme cases and adjust target
            # ================================================================
            unsafe_classes = []
            for label in range(6):
                if label not in indices_by_class:
                    continue

                n_samples = len(indices_by_class[label])
                required_ratio = (
                    target_samples / n_samples if n_samples > 0 else float("inf")
                )

                if required_ratio > max_oversample_ratio:
                    unsafe_classes.append(
                        {
                            "label": label,
                            "name": self.EMOTION_NAMES[label],
                            "samples": n_samples,
                            "ratio": required_ratio,
                        }
                    )

            if unsafe_classes:
                print(
                    f"\n⚠️  SAFETY WARNING: {len(unsafe_classes)} classes exceed safe oversampling ratio!"
                )
                for cls in unsafe_classes:
                    print(
                        f"  Class {cls['label']} ({cls['name']:12s}): {cls['samples']} samples → {cls['ratio']:.1f}x ratio (limit: {max_oversample_ratio}x)"
                    )

                # Calculate SAFE target based on smallest class
                smallest_class_size = min(
                    len(indices_by_class[label]) for label in indices_by_class.keys()
                )
                safe_target = int(smallest_class_size * max_oversample_ratio)

                print(f"\n🛡️  Applying Safety Adjustment:")
                print(f"  Original target: {target_samples}")
                print(
                    f"  Safe target: {safe_target} (based on {smallest_class_size} × {max_oversample_ratio})"
                )

                # Use the safe target instead
                target_samples = min(target_samples, safe_target)

                print(f"  ✓ Using adjusted target: {target_samples}")

            # ================================================================
            # OVERSAMPLE with Safety Limits + Noise Addition (SMOTE-like)
            # ================================================================
            balanced_indices = list(range(n_original_samples))
            additional_samples = []

            print(f"\n📊 Oversampling Plan:")
            for label in range(6):
                if label not in indices_by_class:
                    print(
                        f"  Class {label} ({self.EMOTION_NAMES[label]:12s}): SKIPPED (no samples)"
                    )
                    continue

                class_indices = indices_by_class[label]
                n_samples = len(class_indices)
                n_needed = target_samples - n_samples

                if n_needed > 0:
                    actual_ratio = (n_samples + n_needed) / n_samples

                    # Calculate how many to add
                    n_full_copies = n_needed // n_samples
                    remainder = n_needed % n_samples

                    print(
                        f"  Class {label} ({self.EMOTION_NAMES[label]:12s}): "
                        f"{n_samples:3d} → {target_samples:4d} samples "
                        f"(+{n_needed:4d}, {actual_ratio:.1f}x ratio)"
                    )

                    # Add full copies
                    for _ in range(n_full_copies):
                        additional_samples.extend(class_indices)

                    # Add remainder with random sampling
                    if remainder > 0:
                        additional_samples.extend(
                            np.random.choice(
                                class_indices, remainder, replace=False
                            ).tolist()
                        )
                else:
                    print(
                        f"  Class {label} ({self.EMOTION_NAMES[label]:12s}): "
                        f"{n_samples:3d} samples (no oversampling needed)"
                    )

            # Combine original + additional samples
            shuffled_additional = additional_samples.copy()
            np.random.shuffle(shuffled_additional)
            balanced_indices = list(range(n_original_samples)) + shuffled_additional

            # Final shuffle of all indices
            np.random.shuffle(balanced_indices)

            # ================================================================
            # APPLY SMALL GAUSSIAN NOISE to synthetic samples (SMOTE-like)
            # ================================================================
            print(f"\n🔊 Applying small noise to synthetic samples (SMOTE-like)...")

            balanced_samples = {
                "audio": samples["audio"][balanced_indices].copy(),
                "visual": samples["visual"][balanced_indices].copy(),
                "text": samples["text"][balanced_indices].copy(),
                "labels": samples["labels"][balanced_indices],
                "emotion_scores": samples["emotion_scores"][balanced_indices],
                "video_ids": samples["video_ids"][balanced_indices],
                "segment_ids": samples["segment_ids"][balanced_indices],
            }

            # Add small noise only to synthetic samples (not original ones)
            noise_std = 0.01  # 1% noise
            n_synthetic = len(additional_samples)

            if n_synthetic > 0:
                # Audio noise
                audio_noise = np.random.normal(
                    0, noise_std, balanced_samples["audio"][n_original_samples:].shape
                )
                balanced_samples["audio"][n_original_samples:] += audio_noise

                # Visual noise
                visual_noise = np.random.normal(
                    0, noise_std, balanced_samples["visual"][n_original_samples:].shape
                )
                balanced_samples["visual"][n_original_samples:] += visual_noise

                # Text noise (smaller, as embeddings are more sensitive)
                text_noise = np.random.normal(
                    0,
                    noise_std * 0.5,
                    balanced_samples["text"][n_original_samples:].shape,
                )
                balanced_samples["text"][n_original_samples:] += text_noise

                print(
                    f"  ✓ Added {noise_std*100:.1f}% Gaussian noise to {n_synthetic} synthetic samples"
                )

            # ================================================================
            # REPORT RESULTS
            # ================================================================
            print(f"\n{'='*70}")
            print(f"After Safe Oversampling:")
            print(f"{'='*70}")
            print(
                f"Total samples: {len(balanced_samples['labels'])} (original: {n_original_samples})"
            )

            balanced_counts = Counter(balanced_samples["labels"])
            max_count = max(balanced_counts.values())

            print(f"\nFinal Distribution:")
            for i in range(6):
                count = balanced_counts.get(i, 0)
                original_count = label_counts.get(i, 0)
                added = count - original_count
                pct = (count / len(balanced_samples["labels"])) * 100
                ratio = count / original_count if original_count > 0 else 0

                # Visual indicator of balance
                bar_length = int((count / max_count) * 30)
                bar = "█" * bar_length

                print(
                    f"  Class {i} ({self.EMOTION_NAMES[i]:12s}): "
                    f"{count:4d} samples ({pct:5.2f}%) "
                    f"[+{added:4d}] {ratio:4.1f}x │{bar}"
                )

            # Verify no samples were lost
            unique_original_ids = set(samples["segment_ids"][:n_original_samples])
            unique_balanced_ids_original = set(
                balanced_samples["segment_ids"][:n_original_samples]
            )

            if not unique_original_ids.issubset(unique_balanced_ids_original):
                print(f"\n⚠️  WARNING: Some original samples were lost!")
            else:
                print(
                    f"\n✓ Safety Check: All {n_original_samples} original samples preserved!"
                )

            return balanced_samples

        elif strategy == "undersample":
            print("\n⚠️  WARNING: Undersample strategy will REMOVE samples!")
            print("   This may result in data loss. Are you sure?")

            # Set random seed
            np.random.seed(random_state)

            min_samples = min(label_counts.values())
            print(f"\nTarget samples per class: {min_samples}")

            indices_by_class = defaultdict(list)
            for idx, label in enumerate(labels):
                indices_by_class[label].append(idx)

            balanced_indices = []
            for label in range(6):
                if label not in indices_by_class:
                    print(f"\n⚠️  Skipping class {label} (no samples)")
                    continue

                class_indices = indices_by_class[label]
                n_samples = len(class_indices)

                if n_samples >= min_samples:
                    selected = np.random.choice(
                        class_indices, min_samples, replace=False
                    ).tolist()
                else:
                    # Keep all if less than min_samples
                    selected = class_indices

                balanced_indices.extend(selected)
                print(
                    f"  Class {label} ({self.EMOTION_NAMES[label]:12s}): keeping {len(selected)}/{n_samples} samples"
                )

            np.random.shuffle(balanced_indices)

            balanced_samples = {
                "audio": samples["audio"][balanced_indices],
                "visual": samples["visual"][balanced_indices],
                "text": samples["text"][balanced_indices],
                "labels": samples["labels"][balanced_indices],
                "emotion_scores": samples["emotion_scores"][balanced_indices],
                "video_ids": samples["video_ids"][balanced_indices],
                "segment_ids": samples["segment_ids"][balanced_indices],
            }

            print(f"\nAfter {strategy}:")
            print(
                f"Total samples: {len(balanced_samples['labels'])} (removed: {n_original_samples - len(balanced_samples['labels'])})"
            )
            balanced_counts = Counter(balanced_samples["labels"])
            for i in range(6):
                count = balanced_counts.get(i, 0)
                pct = (
                    (count / len(balanced_samples["labels"])) * 100
                    if len(balanced_samples["labels"]) > 0
                    else 0
                )
                print(
                    f"  Class {i} ({self.EMOTION_NAMES[i]:12s}): {count:5d} samples ({pct:5.2f}%)"
                )

            return balanced_samples

        else:
            print("\n✓ No balancing applied - using original samples as-is")
            return samples

    def save_processed_data(self, samples, output_dir="processed_mosei_data"):
        """Save processed data ke file"""
        print(f"\n{'='*70}")
        print("STEP 3: Saving Processed Data")
        print(f"{'='*70}")

        os.makedirs(output_dir, exist_ok=True)

        # Split data: 70% train, 10% val, 20% test
        indices = np.arange(len(samples["labels"]))
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=samples["labels"]
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.67,
            random_state=42,
            stratify=samples["labels"][temp_idx],
        )

        # Create splits
        splits = {}
        for split_name, split_idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            splits[split_name] = {
                "speech": samples["audio"][
                    split_idx
                ],  # Rename audio -> speech untuk compatibility
                "text": samples["text"][split_idx],
                "visual": samples["visual"][split_idx],
                "labels": samples["labels"][split_idx],
                "emotion_scores": samples["emotion_scores"][split_idx],
                "segment_ids": samples["segment_ids"][split_idx],
            }

            # Save split
            output_file = os.path.join(output_dir, f"{split_name}_data.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(splits[split_name], f)

            print(
                f"✓ Saved {split_name} data: {len(splits[split_name]['labels'])} samples"
            )

        # Save metadata
        metadata = {
            "timestep": self.target_timestep,
            "speech_dim": samples["audio"].shape[2],
            "text_dim": samples["text"].shape[2],
            "visual_dim": samples["visual"].shape[2],
            "n_classes": 6,
            "class_names": self.EMOTION_NAMES,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "threshold": self.threshold,
            "strategy": self.strategy,
        }

        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print(f"\n✓ Metadata saved")
        print(f"\nData shapes:")
        print(f"  Speech: {samples['audio'].shape}")
        print(f"  Text: {samples['text'].shape}")
        print(f"  Visual: {samples['visual'].shape}")
        print(f"  Labels: {samples['labels'].shape}")

        print(f"\n{'='*70}")
        print("PREPROCESSING COMPLETED!")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}/")
        print("Files created:")
        print("  - train_data.pkl")
        print("  - val_data.pkl")
        print("  - test_data.pkl")
        print("  - metadata.pkl")

        return splits, metadata


# Main execution
if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    DATA_DIR = r"data/raw"  # Ganti dengan path Anda
    OUTPUT_DIR = r"data/raw/processed_mosei_data"
    TARGET_TIMESTEP = 20

    # ========================================================================
    # EXPERIMENT 1: Threshold 0.5 (More permissive) with Safe Oversampling
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("🔵 EXPERIMENT 1: Threshold 0.5 (More Permissive) + Safe Oversampling")
    print("=" * 70)

    preprocessor_1 = CMU_MOSEI_Preprocessor(
        data_dir=DATA_DIR,
        target_timestep=TARGET_TIMESTEP,
        threshold=0.5,
        strategy="max",
    )

    try:
        samples_1 = preprocessor_1.load_and_preprocess()
        balanced_samples_1 = preprocessor_1.handle_class_imbalance(
            samples_1,
            strategy="oversample",
            min_samples_per_class=200,
            max_oversample_ratio=5.0,  # Safety limit: max 5x oversampling
            target_mode="median",  # Use median instead of max
        )
        splits_1, metadata_1 = preprocessor_1.save_processed_data(
            balanced_samples_1, output_dir=f"{OUTPUT_DIR}_t05_safe"
        )
    except Exception as e:
        print(f"✗ Experiment 1 failed: {e}")
        import traceback

        traceback.print_exc()

    # ========================================================================
    # EXPERIMENT 2: Threshold 1.0 (Balanced) with Safe Oversampling
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("🟡 EXPERIMENT 2: Threshold 1.0 (Balanced) + Safe Oversampling")
    print("=" * 70)

    preprocessor_2 = CMU_MOSEI_Preprocessor(
        data_dir=DATA_DIR,
        target_timestep=TARGET_TIMESTEP,
        threshold=1.0,
        strategy="max",
    )

    try:
        samples_2 = preprocessor_2.load_and_preprocess()
        balanced_samples_2 = preprocessor_2.handle_class_imbalance(
            samples_2,
            strategy="oversample",
            min_samples_per_class=200,
            max_oversample_ratio=5.0,
            target_mode="median",
        )
        splits_2, metadata_2 = preprocessor_2.save_processed_data(
            balanced_samples_2, output_dir=f"{OUTPUT_DIR}_t10_safe"
        )
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n\n" + "=" * 70)
    print("✓ ALL SAFE EXPERIMENTS COMPLETED!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("1. Review distribusi balanced dengan safety limits")
    print("2. Classes dengan <25 samples akan di-limit ke 5x oversampling")
    print("3. Synthetic samples memiliki small noise untuk diversity")
    print("4. Gunakan class weights saat training untuk fine-tuning")
