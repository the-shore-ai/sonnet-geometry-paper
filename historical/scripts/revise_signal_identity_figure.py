"""Move the identity figure's legend away from data; preserve the first render."""
import json
import matplotlib.pyplot as plt

import summarize_signal_audit as renderer
from analyze_signal_audit import receipt
from sonnet_geometry.storage import canonical_json, write_exclusive


def save_revised(fig, name, values, summary_receipt):
    if name != 'identity-and-relations':
        plt.close(fig)
        return
    fig.axes[0].get_legend().remove()
    # The middle groups are below .6; this legend occupies their empty upper area.
    fig.axes[0].legend(loc='upper center', bbox_to_anchor=(.5, .99), fontsize=9)
    revised = name + '-v2'
    folder = renderer.OUT/'figures'
    for suffix in ('png', 'pdf'):
        target = folder/f'{revised}.{suffix}'
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    write_exclusive(folder/f'{revised}.receipt.json', canonical_json({
        'summary': summary_receipt, 'code': [receipt(__file__), receipt(renderer.__file__)],
        'plotted_values': values, 'change': 'Legend position only; all numerical values unchanged.',
        'previous_render': receipt(folder/f'{name}.png'),
        'bands': 'Dependent five-seed min/max; not confidence intervals.',
        'files': [receipt(folder/f'{revised}.{s}') for s in ('png', 'pdf')]}))


if __name__ == '__main__':
    renderer.save_figure = save_revised
    renderer.figures(json.loads((renderer.OUT/'summary.json').read_text()))
