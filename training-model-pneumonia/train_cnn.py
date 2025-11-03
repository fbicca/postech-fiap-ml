# train_cnn.py
import os, shutil, random, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# -----------------------------
# Configurações
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw"        # opcional: se existir, o script cria o split
TRAIN_DIR= DATA_DIR / "train"
VAL_DIR  = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUT_DIR  = Path("outputs")
MODEL_DIR= OUT_DIR / "models"
PLOTS_DIR= OUT_DIR / "plots"
REPORTS_DIR = OUT_DIR / "reports"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 6
VAL_SPLIT = 0.15
TEST_SPLIT= 0.15

# -----------------------------
# Utils
# -----------------------------
def ensure_dirs():
    for d in [MODEL_DIR, PLOTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def find_classes_from_dir(dir_path: Path):
    classes = [d.name for d in sorted(dir_path.iterdir()) if d.is_dir()]
    return classes

def count_images(base_dir: Path, classes):
    counts = {}
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        counts[split_dir.name] = {c: len(list((split_dir/c).glob("*.*"))) for c in classes}
    return counts

def split_from_raw_if_needed():
    """Se data/train não existir, cria splits a partir de data/raw (pastas por classe)."""
    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print("[split] Pastas de treino/val/test já existem. Pulando divisão.")
        return

    if not RAW_DIR.exists():
        print("[split] data/raw não encontrado e data/train vazio. Forneça dados.")
        return

    # Limpa destinos
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    classes = find_classes_from_dir(RAW_DIR)
    for c in classes:
        files = list((RAW_DIR/c).glob("*.*"))
        random.shuffle(files)
        n = len(files)
        n_test = int(n * TEST_SPLIT)
        n_val  = int(n * VAL_SPLIT)
        n_train= n - n_test - n_val

        splits = {
            TRAIN_DIR/c: files[:n_train],
            VAL_DIR/c:   files[n_train:n_train+n_val],
            TEST_DIR/c:  files[n_train+n_val:]
        }
        for dest, subset in splits.items():
            dest.mkdir(parents=True, exist_ok=True)
            for f in subset:
                shutil.copy2(f, dest)
    print("[split] Divisão criada a partir de data/raw.")

def build_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="categorical",
        image_size=img_size, batch_size=batch_size, seed=SEED, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels="inferred", label_mode="categorical",
        image_size=img_size, batch_size=batch_size, seed=SEED, shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="categorical",
        image_size=img_size, batch_size=batch_size, seed=SEED, shuffle=False
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    # Augmentations simples + preprocess EfficientNet
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ])

    def preprocess(image, label, training=False):
        if training:
            image = data_augmentation(image)
        return preprocess_input(image), label

    train_ds = train_ds.map(lambda x,y: preprocess(x,y,True), num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(lambda x,y: preprocess(x,y,False), num_parallel_calls=AUTOTUNE)
    test_ds  = test_ds.map(lambda x,y: preprocess(x,y,False), num_parallel_calls=AUTOTUNE)

    return (train_ds.cache().prefetch(AUTOTUNE),
            val_ds.cache().prefetch(AUTOTUNE),
            test_ds.cache().prefetch(AUTOTUNE),
            class_names)

def build_model(num_classes):
    base = EfficientNetB0(include_top=False, input_shape=(*IMG_SIZE,3), weights="imagenet")
    base.trainable = False  # fine-tune depois
    inputs = layers.Input(shape=(*IMG_SIZE,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base

def unfreeze_and_finetune(model, base, lr=1e-4):
    base.trainable = True
    # opcional: congele primeiras camadas para estabilidade
    for layer in base.layers[:200]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_confusion(cm, classes, savepath):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Matriz de confusão')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)

def main():
    ensure_dirs()
    split_from_raw_if_needed()

    train_ds, val_ds, test_ds, class_names = build_datasets()
    num_classes = len(class_names)
    print(f"[info] classes: {class_names}")

    # Modelo base (feature extractor)
    model, base = build_model(num_classes)

    callbacks = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=max(2, PATIENCE//2), factor=0.3, monitor="val_loss"),
        ModelCheckpoint(MODEL_DIR/"best_feature_extractor.keras", save_best_only=True,
                        monitor="val_accuracy")
    ]

    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )

    # Fine-tuning (opcional, melhora performance)
    model = unfreeze_and_finetune(model, base, lr=1e-5)
    callbacks_ft = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=max(2, PATIENCE//2), factor=0.3, monitor="val_loss"),
        ModelCheckpoint(MODEL_DIR/"best_finetuned.keras", save_best_only=True,
                        monitor="val_accuracy")
    ]
    hist_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=max(10, EPOCHS//2), callbacks=callbacks_ft
    )

    # Avaliação no TEST
    y_true = []
    y_prob = []
    for batch_imgs, batch_labels in test_ds:
        y_true.append(batch_labels.numpy())
        y_prob.append(model.predict(batch_imgs, verbose=0))
    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    y_pred = y_prob.argmax(axis=1)
    y_true_idx = y_true.argmax(axis=1)

    # Métricas
    report = classification_report(y_true_idx, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(REPORTS_DIR/"classification_report.csv")

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true_idx, y_prob[:,1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = np.nan

    with open(REPORTS_DIR/"summary.json", "w") as f:
        json.dump({
            "classes": class_names,
            "test_samples": int(y_true.shape[0]),
            "AUC": float(auc) if auc==auc else None  # handle NaN
        }, f, indent=2)

    # Matriz de confusão
    cm = confusion_matrix(y_true_idx, y_pred)
    plot_confusion(cm, class_names, PLOTS_DIR/"confusion_matrix.png")

    # Salvar modelo final
    model.save(MODEL_DIR/"model.h5")
    print("[done] Modelo salvo em outputs/models/model.h5")
    print("[done] Relatório em outputs/reports/classification_report.csv")
    print("[done] Matriz de confusão em outputs/plots/confusion_matrix.png")
    print(f"[done] AUC: {auc:.4f}" if auc==auc else "[done] AUC não calculada.")

if __name__ == "__main__":
    main()
