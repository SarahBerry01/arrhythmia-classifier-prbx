import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from random import shuffle
from collections import Counter
import random

from utils import get_signal, get_annotations, get_all_signal_ids
from filters import apply_filters
from segment import get_squared_double_difference, get_peaks
from segment import get_peak_annotation, segment_signal_workflow


def test_show_high_freq_noise():
    signal = get_signal(100)
    filtered = apply_filters(signal)
    figure, ax = plt.subplots(1, 2)
    ax[0].set_title("original")
    ax[1].set_title("filtered")
    ax[0].axhline(y=0, color='black', linestyle='dotted', alpha=0.5)
    ax[1].axhline(y=0, color='black', linestyle='dotted', alpha=0.5)
    ax[0].plot(signal[:500])
    ax[1].plot(filtered[:500])
    plt.tight_layout()
    plt.show()


def test_baseline_wander_removal():
    signal = get_signal(234)
    filtered = apply_filters(signal)
    figure, ax = plt.subplots(2)
    ax[0].set_title("Original")
    ax[1].set_title("Filtered")
    ax[0].axhline(y=0, color='black', alpha=0.2)
    ax[1].axhline(y=0, color='black', alpha=0.2)
    ax[0].plot(signal[:6000], linewidth=0.6)
    ax[1].plot(filtered[:6000], linewidth=0.6)
    x = list(range(6000))
    p = np.poly1d(np.polyfit(x, signal[:6000], 3))
    p_filtered = np.poly1d(np.polyfit(x, filtered[:6000], 3))
    ax[0].plot(p(x))
    ax[1].plot(p_filtered(x))
    plt.tight_layout()
    plt.show()


def test_get_peak_annotation():
    did = 100
    signal = get_signal(did, 0, 1000)
    annotations = get_annotations(did, 0, 1000)
    filtered_signal = apply_filters(signal)
    sdd = get_squared_double_difference(filtered_signal)
    peaks = get_peaks(sdd)
    peaks, annotations = get_peak_annotation(peaks, annotations)
    plt.plot(filtered_signal)
    y = max(filtered_signal)
    for peak, annotation in zip(peaks, annotations):
        plt.annotate(annotation, (peak, y))
    plt.show()


def test_segment_signal():
    did = 102
    signal = get_signal(did)
    annotations = get_annotations(did)
    filtered_signal = apply_filters(signal)
    segments, annotations = segment_signal_workflow(
        filtered_signal, annotations)
    for seg, anno in zip(segments, annotations):
        if anno != 'N':
            plt.plot(seg)
            plt.title(anno)
            plt.show()


def get_peak_detection_rate_no_filter():
    ids = get_all_signal_ids()
    all_annotations = []
    all_detected_annotations = []
    for i, sample_id in enumerate(ids):
        signal = get_signal(sample_id, 0, 3600)
        annotations = get_annotations(sample_id, 0, 3600)
        beat_annotations = [a for a in annotations.symbol if a in [
            'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']]
        all_annotations += beat_annotations
        sdd = get_squared_double_difference(signal)
        peaks = get_peaks(sdd)
        peaks, detected_annotations = get_peak_annotation(peaks, annotations)
        all_detected_annotations += (detected_annotations)
        print(sample_id, len(detected_annotations)/len(beat_annotations))
    print(len(all_detected_annotations), len(all_annotations))
    print(len(all_detected_annotations)/len(all_annotations))
    print(Counter(all_detected_annotations), Counter(all_annotations))


def get_peak_detection_rate():
    ids = get_all_signal_ids()
    all_annotations = []
    all_detected_annotations = []
    for i, sample_id in enumerate(ids):
        signal = get_signal(sample_id, 0, 3600)
        annotations = get_annotations(sample_id, 0, 3600)
        beat_annotations = [a for a in annotations.symbol if a in [
            'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']]
        all_annotations += beat_annotations
        filtered_signal = apply_filters(signal)
        sdd = get_squared_double_difference(filtered_signal)
        peaks = get_peaks(sdd)
        peaks, detected_annotations = get_peak_annotation(peaks, annotations)
        all_detected_annotations += (detected_annotations)
        print(sample_id, len(detected_annotations)/len(beat_annotations))
    print(len(all_detected_annotations), len(all_annotations))
    print(len(all_detected_annotations)/len(all_annotations))
    print(Counter(all_detected_annotations), Counter(all_annotations))


def get_class_distribution_post_process():
    print("== PREPARING SEGMENTS ==")
    ids = get_all_signal_ids()
    all_annotations = []
    for i, sample_id in enumerate(ids):
        print(int(i*100/len(ids)), "%")
        signal = get_signal(sample_id)
        annotations = get_annotations(sample_id)
        filtered_signal = apply_filters(signal)
        segments, segment_annotations = segment_signal_workflow(
            filtered_signal, annotations)
        all_annotations += (segment_annotations)
    values, bins, bars = plt.hist(
        all_annotations, bins=np.arange(15)-0.5, alpha=0.5)
    plt.bar_label(bars, fontsize=8)
    plt.show()


def get_class_distribution_preprocess():
    beat_anno = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V',
                 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    ids = get_all_signal_ids()
    all_annotations = []
    for i, sample_id in enumerate(ids):
        print(int(i*100/len(ids)), "%")
        annotations = get_annotations(sample_id).symbol
        annotations = [a for a in annotations if a in beat_anno]
        all_annotations += (annotations)
        all_annotations += (annotations)
    dist = dict(Counter(all_annotations))
    dist.pop('N')
    dist = dict(sorted(dist.items(), key=lambda item: item[1]))
    bars = plt.barh(list(dist.keys()), list(
        dist.values()), zorder=3, height=0.9)
    plt.tight_layout()
    plt.grid(visible=True, axis="x", zorder=0, which="both")
    plt.show()


def test_sdd():
    for did in [100, 102, 109, 200, 124]:
        print(get_annotations(did, 1000, 3500).symbol)
        signal = get_signal(did, 1000, 2500)
        filtered_signal = apply_filters(signal)
        sdd = get_squared_double_difference(filtered_signal)
        peaks = get_peaks(sdd)
        figure, ax = plt.subplots(2)
        ax[0].plot(filtered_signal)
        for peak in peaks:
            ax[0].axvline(peak, color="orange", linestyle="dashed", alpha=0.8)
        ax[1].plot(sdd)
        plt.show()


def test_polyfit():
    did = 109
    signal = get_signal(did)
    x = range(len(signal))
    poly = np.poly1d(np.polyfit(x, signal, 50))(x)
    plt.plot(signal[:10000])
    plt.plot(poly[:10000])
    plt.show()


if __name__ == '__main__':
    globals()[sys.argv[1]]()
