#!/usr/bin/env python
# coding: utf-8

import time
import pyaudio
from scipy.fft import fft, fftfreq
import numpy as np
import re
from scipy import signal
import random

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 16000

# TODO here and otherwise, reduce use of globals, either shifting away from callback paradigm or to using partials
goal_note = None


def main():
    global goal_note

    loadpitches()

    audio = pyaudio.PyAudio()
    index = return_default_index(audio)
    print("recording via index "+str(index))

    audio.get_device_info_by_index(index)

    peaks, amplitudes = find_recorded_peaks_and_amplitudes(audio, index)

    while True:
        # pick a random note/pitch
        goal_note, goal_pitch = random.choice(list(pitches.items()))
        goal_note = re.sub(r'[0-9]', "", goal_note)
        # print("goal note: {}, goal pitch: {}".format(goal_note, goal_pitch))
        output_data = mimic_sound(peaks, amplitudes, goal_pitch)
        stream = audio.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=RATE,
                            output=True)
        stream.write(output_data.tobytes())
        stream.stop_stream()
        stream.close()

        # stop it from recording itself
        time.sleep(.5)

        # listen for you to play the goal note
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK, stream_callback=find_note_callback)
        while stream.is_active():
            time.sleep(.1)
        stream.stop_stream()
        stream.close()

        # a little reward for getting the note (immediate feedback)
        stream = audio.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=RATE,
                            output=True)
        stream.write(output_data[:len(output_data)//2].tobytes())
        stream.stop_stream()
        stream.close()

        time.sleep(5)


# TODO use binary search
def find_note(pitch, tolerance=15):
    global pitches
    candidates = []
    for note, note_pitch in pitches.items():
        error = abs(pitch - note_pitch)
        if error < tolerance:
            candidates.append((error, note))
    candidates.sort()
    if len(candidates) > 0:
        print("{} {}".format(candidates, pitch))
        return candidates[0][1]
    return None


pitches = None


def loadpitches():
    global pitches
    pitches = {'G3': 196,
               'Gsharp3': 207.65,
               'A3': 220,
               'Asharp3': 233.08,
               'B3': 246.94,
               'C4': 261.63,
               'Csharp4': 277.18,
               'D4': 293.66,
               'Dsharp4': 311.13,
               'E4': 329.63,
               'F4': 349.23,
               'Fsharp4': 369.99,
               'G4': 392,
               'Gsharp4': 415.3,
               'A4': 440,
               'Asharp4': 466.16,
               'B4': 493.88,
               'C5': 523.25,
               'Csharp5': 554.37,
               'D5': 587.33,
               'Dsharp5': 622.25,
               'E5': 659.25,
               'F5': 698.46,
               'Fsharp5': 739.99,
               'G5': 783.99,
               'Gsharp5': 830.61,
               'A5': 880,
               'Asharp5': 932.33,
               'B5': 987.77,
               'C6': 1046.5,
               'Csharp6': 1108.73,
               'D6': 1174.66}


def return_default_index(audio):
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if audio.get_device_info_by_host_api_device_index(0, i).get('name') == 'default':
            return i
    return None


# TODO reduce the use of globals either by dropping the callback paradigm or switching to partials
record_current_note = None
record_note_in_a_row = 0
record_note_dat = np.array([])


def record_note_callback(rawdata, frame_count, time_info, flag):
    global record_note_in_a_row
    global record_current_note
    global record_note_dat

    dat = np.frombuffer(rawdata, dtype=np.int16)
    N = len(dat)
    T = 1.0 / RATE
    yf = np.abs(fft(dat))[:N//2]
    xf = fftfreq(N, T)[:N//2]
    if (np.max(yf) > 700000):
        peaks = signal.find_peaks_cwt(dat, (20,))
        if len(peaks) >= 2:
            peaks_pitch = len(peaks) * RATE / (peaks[-1] - peaks[0])
        else:
            peaks_pitch = 1000000
        # confirm peaks pitch
        fft_pitch = xf[np.argmax(yf)]
        if abs(peaks_pitch - fft_pitch) < 5:
            pitch = np.mean([peaks_pitch, fft_pitch])
        else:
            pitch = None
        note = find_note(pitch, tolerance=15) if pitch is not None else None
        note = re.sub(r'[0-9]', "", note) if note is not None else None
    else:
        note = None

    if (note is not None and note == record_current_note):
        record_note_dat = np.append(record_note_dat, dat)
        record_note_in_a_row += 1
    elif note is None:
        pass
    else:
        record_note_dat = dat
        record_current_note = note
        record_note_in_a_row = 1

    if record_note_in_a_row > 2:
        record_note_in_a_row = 0
        print("you played {}!".format(note))
        return None, pyaudio.paComplete
    return None, pyaudio.paContinue


wrong_note_in_a_row = 0
find_current_note = None
find_note_in_a_row = 0


def find_note_callback(rawdata, frame_count, time_info, flag):
    global wrong_note_in_a_row
    global find_note_in_a_row
    global find_current_note
    global goal_note

    dat = np.frombuffer(rawdata, dtype=np.int16)
    N = len(dat)
    T = 1.0 / RATE
    # TODO try using dst, here and otherwise
    yf = np.abs(fft(dat))[:N//2]
    xf = fftfreq(N, T)[:N//2]
    if (np.max(yf) > 800000):
        loud_tones = np.argsort(yf)[:2]
        # we are having a problem picking the fifth, so if there are two comparable tones, pick the lower
        ratio = loud_tones[0] / loud_tones[1]
        if ratio > 2:
            pitch = xf[loud_tones[0]]
        elif ratio < .5:
            pitch = xf[loud_tones[1]]
        else:
            # pick the lower one
            pitch = np.min(xf[loud_tones])
        note = find_note(xf[np.argmax(yf)], tolerance=15)
        note = re.sub(r'[0-9]', "", note) if note is not None else None
    else:
        note = None

    if (note is not None and note == find_current_note):
        find_note_in_a_row += 1
    elif note is None:
        pass
    else:
        find_current_note = note
        find_note_in_a_row = 1

    if find_note_in_a_row > 1:
        find_note_in_a_row = 0
        print("you played {}!".format(note))
        if goal_note == note:
            print("you matched the goal note!!")
            wrong_note_in_a_row = 0
            return None, pyaudio.paComplete
        else:
            wrong_note_in_a_row += 1
            if wrong_note_in_a_row > 3:
                wrong_note_in_a_row = 0
                print("goal note is {}".format(goal_note))

    return None, pyaudio.paContinue

# inspired by https://mapio.github.io/sinuous-violin/


def find_recorded_peaks_and_amplitudes(audio, index):
    global record_note_dat

    # TODO arg controls if you have to record or not
    print("play your note")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK, stream_callback=record_note_callback)
    while stream.is_active():
        time.sleep(0.1)
    stream.stop_stream()
    stream.close()

    N = len(record_note_dat)
    T = 1.0 / RATE
    record_yf = np.abs(fft(record_note_dat))[:N//2]
    record_xf = fftfreq(N, T)[:N//2]

    peak_indices = signal.find_peaks_cwt(record_yf, widths=(30,))
    amplitudes_maxima = list(map(lambda idx: np.max(
        record_yf[idx - 10:idx + 10]), peak_indices))

    # TODO this could be a bit more elegant in case of duplicate values
    frequencies_maxima = record_xf[np.isin(
        record_yf, amplitudes_maxima)].astype(np.float32)
    amplitudes_maxima = (np.array(amplitudes_maxima) /
                         np.max(amplitudes_maxima)).astype(np.float32)
    return frequencies_maxima, amplitudes_maxima


def sine_wave(freq, duration):
    global RATE
    return np.sin(np.arange(0, duration, 1 / RATE) * freq * 2 * np.pi).astype(np.float32)

# TODO try using ifft


def mimic_sound(frequencies, amplitudes, goal_pitch):
    return sum(map(lambda fa: sine_wave(fa[0] * goal_pitch / frequencies[np.argmax(amplitudes)], 2) * fa[1], zip(frequencies, amplitudes))).astype(np.float32)


main()
