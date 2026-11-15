import matplotlib.pyplot as plt

# Fitted MSD exponents from the five selected trajectories in Figure 3
vesicles = ['9', '11', '13', '2+23', '33']
alpha = [0.64, 0.90, 1.98, 1.64, 1.23]

# Approximate normal range: 0.90 to 1.10 (a descriptive cutoff)
groups = ['Subdiffusive', 'Approximately normal', 'Superdiffusive']
colours = [
    ['royalblue', 'dodgerblue', 'deepskyblue'],
    ['goldenrod', 'darkorange', 'orange'],
    ['firebrick', 'red', 'lightcoral']
]

fig, ax = plt.subplots(figsize=(7, 4.5))

for group in range(3):
    # Find the vesicles in this diffusion category
    points = []
    for name, value in zip(vesicles, alpha):
        if value < 0.90:
            category = 0
        elif value <= 1.10:
            category = 1
        else:
            category = 2
        if category == group:
            points.append((name, value))

    if not points:
        continue

    # Points are spread slightly so vesicles do not overlap
    for i, (name, value) in enumerate(points):
        x = group + (i - (len(points) - 1) / 2) * 0.15
        ax.scatter(x, value, color=colours[group][i % 3], s=70, zorder=3)
        ax.annotate(name, (x, value), xytext=(5, 6),
                    textcoords='offset points', fontsize=9)

    # Short dashed black line = category mean
    mean = sum(value for _, value in points) / len(points)
    ax.hlines(mean, group - 0.28, group + 0.28,
              color='black', linestyle='--', linewidth=1.8)
    print(f'{groups[group]}: mean alpha = {mean:.2f} (n={len(points)})')

# Solid black reference = normal Brownian diffusion
ax.axhline(1, color='black', linestyle='-', linewidth=1.2,
           label='Brownian reference (α = 1)')
ax.plot([], [], color='black', linestyle='--',
        label='Category mean')

# Match the styling of the earlier lab figures
ax.text(-0.10, 1.02, 'A', transform=ax.transAxes,
        fontsize=24, fontweight='bold')
ax.set_title('Clustering analysis', loc='left', fontsize=12)
ax.set_xlabel('Diffusion category')
ax.set_ylabel('MSD scaling exponent (α)')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(groups)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 2.3)
ax.legend(frameon=False, loc='upper left', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Figure_4_clustering.pdf', bbox_inches='tight')
# plt.savefig('Figure_4_clustering.png', dpi=300, bbox_inches='tight')
plt.show()
