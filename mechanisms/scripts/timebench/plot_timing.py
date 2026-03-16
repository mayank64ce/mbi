#!/usr/bin/env python3
"""
Generate visualizations for FHAIM timing data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from logs
datasets = ['COMPAS', 'Cancer', 'Diabetes']

# L1 data
l1_data = {
    'compute_1way': [24.8, 30.7, 32.0],
    'compute_2way': [292, 682, 622],
    'select': [59, 129, 101],
    'gumbel': [3.9, 4.8, 3.9],
    'measure': [0.34, 0.28, 0.40],
}

# L2 data
l2_data = {
    'compute_1way': [24.7, 30.6, 32.0],
    'compute_2way': [293, 680, 628],
    'select': [70, 140, 112],
    'gumbel': [4.0, 4.8, 4.0],
    'measure': [0.70, 1.18, 2.25],
}


def plot_stacked_bar():
    """Stacked bar chart showing time breakdown per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(datasets))
    width = 0.6

    for idx, (data, title) in enumerate([(l1_data, 'FHAIM-L1'), (l2_data, 'FHAIM-L2')]):
        ax = axes[idx]

        # Stack the bars
        bottom = np.zeros(3)

        colors = ['#2ecc71', '#27ae60', '#3498db', '#9b59b6', '#e74c3c']
        labels = ['1-way Compute', '2-way Compute', 'Select', 'Gumbel', 'Measure']

        for i, (key, label, color) in enumerate(zip(
            ['compute_1way', 'compute_2way', 'select', 'gumbel', 'measure'],
            labels, colors
        )):
            values = data[key]
            ax.bar(x, values, width, label=label, bottom=bottom, color=color)
            bottom += values

        ax.set_ylabel('Time (seconds)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('timing_stacked_bar.pdf', bbox_inches='tight')
    plt.savefig('timing_stacked_bar.png', dpi=200, bbox_inches='tight')
    print("Saved: timing_stacked_bar.pdf/png")


def plot_grouped_bar():
    """Grouped bar chart comparing L1 vs L2."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = np.arange(len(datasets))
    width = 0.35

    # Plot 1: Compute time
    ax = axes[0]
    l1_compute = [a + b for a, b in zip(l1_data['compute_1way'], l1_data['compute_2way'])]
    l2_compute = [a + b for a, b in zip(l2_data['compute_1way'], l2_data['compute_2way'])]
    ax.bar(x - width/2, l1_compute, width, label='L1', color='#3498db')
    ax.bar(x + width/2, l2_compute, width, label='L2', color='#e74c3c')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Compute')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # Plot 2: Select time
    ax = axes[1]
    ax.bar(x - width/2, l1_data['select'], width, label='L1', color='#3498db')
    ax.bar(x + width/2, l2_data['select'], width, label='L2', color='#e74c3c')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Select (avg per iteration)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # Plot 3: Measure time
    ax = axes[2]
    ax.bar(x - width/2, l1_data['measure'], width, label='L1', color='#3498db')
    ax.bar(x + width/2, l2_data['measure'], width, label='L2', color='#e74c3c')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Measure (avg per iteration)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.tight_layout()
    plt.savefig('timing_grouped_bar.pdf', bbox_inches='tight')
    plt.savefig('timing_grouped_bar.png', dpi=200, bbox_inches='tight')
    print("Saved: timing_grouped_bar.pdf/png")


def plot_pie_chart():
    """Pie charts showing time distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2ecc71', '#27ae60', '#3498db', '#9b59b6', '#e74c3c']
    labels = ['1-way', '2-way', 'Select', 'Gumbel', 'Measure']

    for idx, (data, title) in enumerate([(l1_data, 'FHAIM-L1'), (l2_data, 'FHAIM-L2')]):
        ax = axes[idx]

        # Average across datasets
        values = [
            np.mean(data['compute_1way']),
            np.mean(data['compute_2way']),
            np.mean(data['select']),
            np.mean(data['gumbel']),
            np.mean(data['measure']),
        ]

        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig('timing_pie.pdf', bbox_inches='tight')
    plt.savefig('timing_pie.png', dpi=200, bbox_inches='tight')
    print("Saved: timing_pie.pdf/png")


def plot_horizontal_bar():
    """Horizontal bar chart - compact and paper-friendly."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        'COMPAS (L1)', 'COMPAS (L2)',
        'Breast (L1)', 'Breast (L2)',
        'Diabetes (L1)', 'Diabetes (L2)',
    ]

    y = np.arange(len(categories))

    # Combine data
    compute_1way = l1_data['compute_1way'] + l2_data['compute_1way']
    compute_1way = [l1_data['compute_1way'][0], l2_data['compute_1way'][0],
                    l1_data['compute_1way'][1], l2_data['compute_1way'][1],
                    l1_data['compute_1way'][2], l2_data['compute_1way'][2]]

    compute_2way = [l1_data['compute_2way'][0], l2_data['compute_2way'][0],
                    l1_data['compute_2way'][1], l2_data['compute_2way'][1],
                    l1_data['compute_2way'][2], l2_data['compute_2way'][2]]

    select = [l1_data['select'][0], l2_data['select'][0],
              l1_data['select'][1], l2_data['select'][1],
              l1_data['select'][2], l2_data['select'][2]]

    gumbel = [l1_data['gumbel'][0], l2_data['gumbel'][0],
              l1_data['gumbel'][1], l2_data['gumbel'][1],
              l1_data['gumbel'][2], l2_data['gumbel'][2]]

    measure = [l1_data['measure'][0], l2_data['measure'][0],
               l1_data['measure'][1], l2_data['measure'][1],
               l1_data['measure'][2], l2_data['measure'][2]]

    # Plot horizontal stacked bars
    left = np.zeros(len(categories))

    colors = ['#2ecc71', '#27ae60', '#3498db', '#9b59b6', '#e74c3c']
    labels = ['1-way Compute', '2-way Compute', 'Select', 'Gumbel', 'Measure']

    for values, label, color in zip(
        [compute_1way, compute_2way, select, gumbel, measure],
        labels, colors
    ):
        ax.barh(y, values, left=left, label=label, color=color, height=0.7)
        left = [l + v for l, v in zip(left, values)]

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('FHAIM Runtime Breakdown')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('timing_horizontal.pdf', bbox_inches='tight')
    plt.savefig('timing_horizontal.png', dpi=200, bbox_inches='tight')
    print("Saved: timing_horizontal.pdf/png")


def plot_compute_breakdown():
    """Focus on compute time breakdown (dominates total time)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(datasets))
    width = 0.35

    # L1 stacked
    ax.bar(x - width/2, l1_data['compute_1way'], width, label='1-way (L1)', color='#3498db')
    ax.bar(x - width/2, l1_data['compute_2way'], width, bottom=l1_data['compute_1way'],
           label='2-way (L1)', color='#2980b9')

    # L2 stacked
    ax.bar(x + width/2, l2_data['compute_1way'], width, label='1-way (L2)', color='#e74c3c')
    ax.bar(x + width/2, l2_data['compute_2way'], width, bottom=l2_data['compute_1way'],
           label='2-way (L2)', color='#c0392b')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Compute Phase Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # Add percentage labels
    for i, dataset in enumerate(datasets):
        total_l1 = l1_data['compute_1way'][i] + l1_data['compute_2way'][i]
        pct_2way = l1_data['compute_2way'][i] / total_l1 * 100
        ax.annotate(f'{pct_2way:.0f}%', xy=(i - width/2, total_l1 + 10), ha='center', fontsize=12)

        total_l2 = l2_data['compute_1way'][i] + l2_data['compute_2way'][i]
        pct_2way = l2_data['compute_2way'][i] / total_l2 * 100
        ax.annotate(f'{pct_2way:.0f}%', xy=(i + width/2, total_l2 + 10), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('timing_compute.pdf', bbox_inches='tight')
    plt.savefig('timing_compute.png', dpi=200, bbox_inches='tight')
    print("Saved: timing_compute.pdf/png")


if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')

    # Increase font sizes for better visibility
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
    })

    plot_stacked_bar()
    plot_grouped_bar()
    plot_pie_chart()
    plot_horizontal_bar()
    plot_compute_breakdown()

    print("\nAll plots saved!")
