"""
Generate the four figures used in the report.

"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def distance_metrics(results_md=None):
    """Return 60-minute MAE by nearest-venue bin, speed-only then full model."""
    if results_md is None:
        # RESULTS.md: 'Stratified by Distance from Venue (60 min horizon)'.
        # These are reported model scores, not estimates from the raw EDA data.
        return (np.array([2.286, 2.258, 2.573, 2.384, 2.176]),
                np.array([1.957, 1.977, 2.236, 2.067, 1.881]))
    text = Path(results_md).read_text(encoding='utf-8')
    section = text.split(
        '## Stratified by Distance from Venue (60 min horizon)', 1)[1]
    section = section.split('### Proximity effect', 1)[0]
    scores = {}
    for line in section.splitlines():
        cells = [cell.strip().replace('**', '')
                 for cell in line.strip('|').split('|')]
        if cells[0] in ('0_speed', '6_all'):
            scores[cells[0]] = np.array([float(v) for v in cells[1:]])
    if set(scores) != {'0_speed', '6_all'} or any(v.shape != (5,) for v in scores.values()):
        raise ValueError(
            'RESULTS.md must contain five distance-bin MAEs for 0_speed and 6_all')
    return scores['0_speed'], scores['6_all']


def generate_figures(project_root, output_dir=None, sensor_metadata=None, results_md=None):

    project_root = Path(project_root).resolve()
    raw = project_root / 'data/raw'
    out = Path(output_dir) if output_dir is not None else project_root / \
        'results/figures'
    sensor_path = (Path(sensor_metadata) if sensor_metadata is not None else
                   raw / 'augmented-pems-bay/data/sensor_graph/sensor_metadata.csv')
    if not sensor_path.is_file():
        raise FileNotFoundError(
            f'{sensor_path} is missing. Supply --sensor-metadata with the downloaded PEMS-BAY sensor_metadata.csv path.')
    events = pd.read_csv(raw/'events/events.csv',
                         parse_dates=['start_time', 'end_time'])
    venues = pd.read_csv(raw/'events/venues.csv')
    weather = pd.read_csv(raw/'weather/weather_5min.csv',
                          parse_dates=['timestamp']).set_index('timestamp')
    sensors = pd.read_csv(sensor_path)
    # Validate optional result data before writing any figures.
    distance_metrics(results_md)
    idx = weather.index
    ids = np.arange(len(idx)-23)
    gap = np.flatnonzero(np.diff(idx.values).astype(
        'timedelta64[m]').astype(int) != 5)
    good = np.ones(len(ids), bool)
    for j in gap:
        good[max(0, j-22):j+1] = False
    ids = ids[good]
    assert len(ids) == 52070
    val_start = idx[ids[int(len(ids)*.7)]+12]
    test_ids = ids[int(len(ids)*.8):]+12
    test_start = idx[test_ids[0]]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        fig.savefig(out/(name+'.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.8))
    ax.scatter(sensors.Longitude, sensors.Latitude, s=8,
               c='#3478ad', alpha=.65, label='PEMS-BAY sensors')
    for i, v in venues.iterrows():
        ax.scatter(v.venue_lon, v.venue_lat, marker='*', s=190,
                   c='#cf6b32', edgecolors='white', linewidth=.5, zorder=5)
        ax.annotate(str(i+1), (v.venue_lon, v.venue_lat),
                    xytext=(7, 7), textcoords='offset points', weight='bold')
    for name, lat, lon in [('SJC', 37.3626, -121.9291), ('NUQ', 37.4161, -122.0494)]:
        ax.scatter(lon, lat, marker='^', s=65, c='#49794b', zorder=6)
        ax.annotate(name, (lon, lat), xytext=((-29, 3) if name ==
                    'SJC' else (6, -10)), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Longitude', ylabel='Latitude')
    ax.set_aspect(1/np.cos(np.deg2rad(37.35)))

    import contextily as ctx
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    basemap = out / 'fig1_osm_basemap.tif'
    if not basemap.exists():
        ctx.bounds2raster(west, south, east, north, str(basemap),
                          zoom=12, ll=True, source=ctx.providers.OpenStreetMap.Mapnik,
                          headers={
                              'User-Agent': 'TrafficShockwaveReport/1.0 (academic figure generation)'},
                          timeout=30)
    ctx.add_basemap(ax, source=str(basemap), crs='EPSG:4326',
                    attribution='(C) OpenStreetMap contributors',
                    attribution_size=5, alpha=0.65)
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=8)
    venue_labels = [f'{i+1} {v}' for i, v in enumerate(venues.venue_name)]
    fig.text(.14, -.035, '     '.join(venue_labels[:2]) +
             '\n' + '     '.join(venue_labels[2:]), fontsize=9)
    save(fig, 'fig1_map')

    def category(t):
        if t in ['NHL', 'AHL', 'MLS']:
            return t
        if 'concert' in t or 'festival' in t:
            return 'Concert / festival'
        return 'Other'
    events['category'] = events.event_type.map(category)
    order = venues.sort_values(
        'venue_lat', ascending=False).venue_name.tolist()
    palette = {'NHL': '#3175a8', 'AHL': '#7aa5c9', 'MLS': '#47956b',
               'Concert / festival': '#cf793c', 'Other': '#9277ae'}
    fig, ax = plt.subplots(figsize=(10, 2.75))
    for cat, col in palette.items():
        e = events[events.category == cat]
        ax.scatter(e.start_time, [order.index(v) for v in e.venue_name], s=e.expected_attendance /
                   380, c=col, alpha=.8, edgecolors='white', linewidth=.4, label=cat)
    ax.set_yticks(range(4), order)
    ax.invert_yaxis()
    ax.axvline(val_start, c='.4', ls='--', lw=1)
    ax.axvline(test_start, c='.4', ls='--', lw=1)
    ax.axvspan(test_start, idx[-1], color='#e5edf4', zorder=-1)
    ax.set_xlim(idx[0]-pd.Timedelta(days=2), idx[-1]+pd.Timedelta(days=2))
    ax.set_ylim(3.6, -.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(axis='y', alpha=.15)
    ax.legend(ncol=5, loc='upper center',
              bbox_to_anchor=(.5, 1.27), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig2_event_timeline')

    daily = weather.resample('D')
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 5.3), sharex=True)
    axes[0].fill_between(daily.mean().index, daily.temperature.min(
    ), daily.temperature.max(), alpha=.2, color='#ca723b')
    axes[0].plot(daily.temperature.mean(), lw=1, color='#b36432')
    axes[0].set_ylabel('Temperature (°C)')
    axes[1].bar(daily.precipitation.max().index,
                daily.precipitation.max(), width=1, color='#3478ad')
    axes[1].set_ylabel('Max. precip.\nreading (mm)')
    fraction = (weather.precipitation >= .51).resample('D').mean()*100
    axes[2].bar(fraction.index, fraction, width=1, color='#49794b')
    axes[2].set_ylabel('Adverse steps (%)')
    for ax in axes:
        ax.axvspan(test_start, idx[-1], color='#e5edf4', zorder=-2)
        ax.grid(axis='y', alpha=.15)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].set_xlim(idx[0], idx[-1])
    fig.tight_layout()
    save(fig, 'fig4_weather')

    baseline, full = distance_metrics(results_md)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(5)
    ax.plot(x, baseline, 'o-', c='#7b8793', label='Speed only')
    ax.plot(x, full, 's-', c='#27739c', label='All variables')
    ax.set_xticks(x, ['0–1\n(n=10)', '1–2\n(n=35)',
                  '2–5\n(n=132)', '5–10\n(n=122)', '10–20\n(n=26)'])
    ax.set(xlabel='Distance to nearest venue (km)',
           ylabel='60-minute MAE (mph)', ylim=(1.7, 2.75))
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save(fig, 'fig5_distance')

    egress = np.zeros(len(idx), bool)
    for end in events.loc[events.expected_attendance >= 15000, 'end_time']:
        egress |= (idx >= end) & (idx < end+pd.Timedelta(minutes=60))
    holidays = np.isin(idx.strftime(
        '%Y-%m-%d'), ['2017-01-01', '2017-01-02', '2017-01-16', '2017-02-20', '2017-05-29'])
    adverse = weather.precipitation.to_numpy() >= .51
    counts = {'egress': int(egress[test_ids].sum()), 'holiday': int(holidays[test_ids].sum()), 'weather': int(
        adverse[test_ids].sum()), 'normal': int((~(egress | holidays | adverse))[test_ids].sum())}
    assert counts == {'egress': 234, 'holiday': 288,
                      'weather': 0, 'normal': 9892}, counts

    lat = np.deg2rad(sensors.Latitude.to_numpy())[:, None]
    lon = np.deg2rad(sensors.Longitude.to_numpy())[:, None]
    vlat = np.deg2rad(venues.venue_lat.to_numpy())[None, :]
    vlon = np.deg2rad(venues.venue_lon.to_numpy())[None, :]
    dist = 6371*2*np.arcsin(np.sqrt(np.sin((lat-vlat)/2) **
                            2+np.cos(lat)*np.cos(vlat)*np.sin((lon-vlon)/2)**2))
    nodes = np.histogram(dist.min(axis=1), [0, 1, 2, 5, 10, 20])[0].tolist()
    assert nodes == [10, 35, 132, 122, 26], nodes
    summary = {'events': len(events), 'validation_first_target': str(val_start), 'test_first_target': str(test_start), 'test_counts': counts,
               'distance_bin_counts': nodes, 'adverse_steps_full_period': int(adverse.sum()), 'distance_mae_reduction_pct': ((baseline-full)/baseline*100).round(2).tolist()}
    (out/'verified_data_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # src/eda.py works from any cwd; the loose transfer copy defaults to cwd.
    default_root = Path(__file__).resolve().parents[1] if Path(
        __file__).parent.name == 'src' else Path.cwd()
    parser.add_argument('--project-root', type=Path, default=default_root)
    parser.add_argument('--output-dir', type=Path,
                        help='Default: PROJECT_ROOT/results/figures')
    parser.add_argument('--sensor-metadata', type=Path,
                        help='Override the PEMS-BAY sensor_metadata.csv location')
    parser.add_argument('--results-md', type=Path,
                        help='Read distance scores from the modeling RESULTS.md')
    args = parser.parse_args(argv)
    generate_figures(args.project_root, args.output_dir,
                     args.sensor_metadata, args.results_md)


if __name__ == '__main__':
    main()
