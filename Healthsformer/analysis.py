import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# loading saved data
encoding = {'ED_intime': 0, 'ED_outtime': 1, 'CT_time': 2, 'SW_time': 3, 'TPA_time': 4, 'admission_time': 5,
            'discharge_time': 6, 'icuin': 7, 'icuout': 8}
decoding = {0: 'ED_intime', 1: 'ED_outtime', 2: 'CT_time', 3: 'SW_time', 4: 'TPA_time', 5: 'admission_time',
            6: 'discharge_time', 7: 'icuin', 8: 'icuout'}
def load(sequence_size=None):
    if sequence_size is None:
        event_sequences, tdeltas, sex_age = joblib.load("PreparedData/_raw_old_full_sequences.pkl")
    elif sequence_size == 4:
        event_sequences, tdeltas, sex_age = joblib.load("PreparedData/_raw_old_4+_sequences.pkl")
    elif sequence_size == 6:
        event_sequences, tdeltas, sex_age = joblib.load("PreparedData/_raw_old_6+_sequences.pkl")
    else:
        event_sequences, tdeltas, sex_age = joblib.load("PreparedData/_raw_old_8+_sequences.pkl")
    return event_sequences, tdeltas, sex_age


def encode(event_sequences):
    for i, events in enumerate(event_sequences):
        event_sequences[i] = [encoding[event] for event in events]
    return event_sequences


def decode(event_sequences):
    for i, events in enumerate(event_sequences):
        event_sequences[i] = [decoding[event] for event in events]
    return event_sequences


def event_occurrence(event_sequences):
    if isinstance(event_sequences[0][0], str):
        event_sequences = encode(event_sequences)

    occurrence = np.zeros(9)
    for events in event_sequences:
        for j in np.unique(np.array(events)):
            occurrence[j] += 1
    df = pd.DataFrame(zip(['Record count'] + list(encoding.keys()), [len(event_sequences)] + occurrence.tolist()))
    df['Percentages'] = df[1]/len(event_sequences)*100
    print(df)


def eae_distribution(event_sequences):
    """event after event distribution"""
    if isinstance(event_sequences[0][0], str):
        event_sequences = encode(event_sequences)

    counts = np.zeros((9, 9))
    for events in event_sequences:
        for i in range(len(events) - 1):
            counts[events[i], events[i + 1]] += 1

    probs = counts / counts.sum(axis=1).reshape(-1, 1)
    probs = pd.DataFrame(probs).rename(columns=dict(enumerate(encoding.keys())), index=dict(enumerate(encoding.keys())))
    counts = pd.DataFrame(counts).rename(columns=dict(enumerate(encoding.keys())),
                                         index=dict(enumerate(encoding.keys())))

    plt.figure()
    sns.heatmap(probs, annot=True, fmt='.2f')
    plt.ylabel("First event")
    plt.xlabel("Following event")
    plt.title("Probability distribution of one event after another")
    plt.show()


def pos_distribution(event_sequences):
    """Probability distribution of events over sequence positions"""
    if isinstance(event_sequences[0][0], str):
        event_sequences = encode(event_sequences)

    max_seq_length = max([len(events) for events in event_sequences])
    positions = np.zeros((9, max_seq_length))
    for events in event_sequences:
        for i in range(len(events)):
            positions[events[i], i] += 1

    probs = positions / positions.sum(axis=0)
    probs = pd.DataFrame(probs).rename(index=dict(enumerate(encoding.keys())))

    positions = pd.DataFrame(positions).rename(index=dict(enumerate(encoding.keys())))

    plt.figure(figsize=(24, 6))
    sns.heatmap(probs, annot=True, fmt='.2f')
    plt.xlabel("Positions in a sequence")
    plt.ylabel("Event")
    plt.title("Probability distribution of events over sequence positions")
    plt.show()


def stay_duration(event_sequences, tdeltas):
    if isinstance(event_sequences[0][0], str):
        event_sequences = encode(event_sequences)

    ed_periods = []
    for i, events in enumerate(event_sequences):
        if 0 in events and 1 in events:
            ed_periods.append(sum([tdeltas[i][j] for j, event in enumerate(events) if events.index(0) < j <= events.index(1)])/60)
    icu_periods = []
    for i, events in enumerate(event_sequences):
        if 7 in events and 8 in events:
            icu_periods.append(sum([tdeltas[i][j] for j, event in enumerate(events) if events.index(7) < j <= events.index(8)])/60)
    hosp_periods = []
    for i, events in enumerate(event_sequences):
        if 5 in events and 6 in events:
            hosp_periods.append(sum([tdeltas[i][j] for j, event in enumerate(events) if events.index(5) < j <= events.index(6)])/60)

    print("Stay durations:")
    try:
        print(f"Emergency department average stay duration = {sum(ed_periods) / len(ed_periods):.2f} minutes.")
    except ZeroDivisionError:
        print("No intervals for event = Emergency Department found.")
    try:
        print(f"ICU average stay duration = {sum(icu_periods) / len(icu_periods):.2f} minutes.")
    except ZeroDivisionError:
        print("No intervals for event = ICU found.")
    try:
        print(f"Hospital average stay duration = {sum(hosp_periods) / len(hosp_periods):.2f} minutes.")
    except ZeroDivisionError:
        print("No intervals for event = Admission found.")


def plot_histogram(event_sequences, tdeltas, target_event):
    """returns time intervals in MINUTES"""
    periods = []
    if target_event == "icu":
        start = 7
        end = 8
    elif target_event == "emergency":
        start = 0
        end = 1
    elif target_event == "hospital":
        start = 5
        end = 6
    else:
        raise ValueError("Target event must be either 'icu', 'emergency' or 'hospital'")
    if isinstance(event_sequences[0][0], str):
        event_sequences = encode(event_sequences)

    for i, events in enumerate(event_sequences):
        if start in events and end in events:
            periods.append(sum([tdeltas[i][j] for j, event in enumerate(events) if events.index(start) < j <= events.index(end)])/60)
    try:
        total_range = max(periods) - min(periods)
    except ValueError:
        print(f"No intervals for event = {target_event} found.")
        return
    # 10-90 breadth
    periods.sort()
    breadth = max(periods[len(periods)*10 // 100:1+(len(periods) * 90) // 100]) - min(periods[len(periods)*10 // 100:1+(len(periods) * 90) // 100])

    plt.figure(layout='tight', figsize=(8, 6))
    plt.grid(zorder=0)
    sns.histplot(x=periods, stat='density', bins=80, zorder=2)  # each bin represents 1% of the total data
    plt.axvline(x=np.array(periods).mean(), color='red', linestyle='--', label=f'Mean = {np.array(periods).mean():.2f}')
    plt.title(f"Histogram of {target_event} stay durations")
    plt.xlabel(f"Time (minutes)")
    plt.ylabel("Density")
    plt.legend(title=f"Observations = {len(periods)}/{len(event_sequences)} \nTotal value range = {total_range:.0f}\n10-90 percentile range = {breadth:.0f}")
    plt.show()


def time_until_event(event_sequences, tdeltas, target_event):
    """returns time intervals in MINUTES"""
    periods = []
    if target_event == "icu":
        start = 7
    elif target_event == "emergency":
        start = 0
    elif target_event == "hospital":
        start = 5
    elif target_event == "tpa":
        start = 4
    elif target_event == "sw":
        start = 3
    elif target_event == "ct":
        start = 2
    else:
        raise ValueError("Invalid target event")
    if isinstance(event_sequences[0][0], str):
        event_sequences = encode(event_sequences)

    for i, events in enumerate(event_sequences):
        if start in events:
            periods.append(sum([tdeltas[i][j] for j, event in enumerate(events) if events.index(start) <= j])/60)

    if len(periods) == 0:
        print(f"No intervals for event = {target_event} found.")
        return

    plt.figure(layout='tight')
    plt.grid(zorder=0)
    sns.histplot(x=periods, stat='density', bins=80, zorder=2)
    plt.axvline(x=np.array(periods).mean(), color='red', linestyle='--', label=f'Mean = {np.array(periods).mean():.2f}')
    plt.title(f"Histogram of time elapsed until {target_event if target_event != 'hospital' else 'hospital admission'}")
    plt.xlabel(f"Time (minutes)")
    plt.ylabel("Density")
    plt.legend(title=f"Event count = {len(periods)}/{len(event_sequences)}")
    plt.show()


if __name__ == "__main__":
    sequence_size = None  # (full=)None , 4, 6, 8
    event_sequences, tdeltas, sex_age = load(sequence_size=sequence_size)
    plot_histogram(event_sequences, tdeltas, 'emergency')
