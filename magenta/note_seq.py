import mido
import numpy as np


GROOVE_MIDI_DATASET_ROLAND_MAPPING = {
    # https://magenta.tensorflow.org/datasets/groove#drum-mapping
    36: "Kick",
    38: "Snare (Head)",
    40: "Snare (Rim)",
    37: "Snare X-Stick",
    48: "Tom 1",
    50: "Tom 1 (Rim)",
    45: "Tom 2",
    47: "Tom 2 (Rim)",
    43: "Tom 3 (Head)",
    58: "Tom 3 (Rim)",
    46: "HH Open (Bow)",
    26: "HH Open (Edge)",
    42: "HH Closed (Bow)",
    22: "HH Closed (Edge)",
    44: "HH Pedal",
    49: "Crash 1 (Bow)",
    55: "Crash 1 (Edge)",
    57: "Crash 2 (Bow)",
    52: "Crash 2 (Edge)",
    51: "Ride (Bow)",
    59: "Ride (Edge)",
    53: "Ride (Bell)"
}


class NoteSequences:
    def __init__(self, fname, downsample_ticks_num=3, note_range=128, **kwargs):
        mid_obj = mido.MidiFile(fname)
        
        self.mido_type = mid_obj.type
        assert self.mido_type == 0, "single track 이외 type에 대하여 아직 구현되지 않았습니다."

        self.org_ticks_per_beat = mid_obj.ticks_per_beat
        self.dwn_ticks_per_beat = downsample_ticks_num
        assert self.org_ticks_per_beat % self.dwn_ticks_per_beat == 0

        self.note_range = note_range
        if self.note_range != 128:
            self.mapping_idx = kwargs["mapping_idx"]

        self.tracks = list()
        for t in mid_obj.tracks:
            self.tracks.append(self.read_track(t))
    
    def _note_to_pitch_index(self, note):
        if self.note_range != 128:
            return self.mapping_idx.index(note)
        else:
            return note
    
    def _pitch_index_to_note(self, idx):
        if self.note_range != 128:
            return self.mapping_idx[idx]
        else:
            return idx
    
    def read_track(self, track):
        numerator, denominator = 4, 4 # default
        downsample_tick_rate = self.org_ticks_per_beat // (self.dwn_ticks_per_beat * numerator * denominator)

        bar = None
        time = 0 # ticks 단위
        track_list = list() # bar 단위로 악보를 저장합니다.
        
        for m in track:
            time += m.time
            if time >= self.org_ticks_per_beat:
                time -= self.org_ticks_per_beat
                if bar is not None:
                    track_list.append(bar)
                    bar = None
            
            if type(m) is mido.messages.messages.Message:
                # note_on / note_off 이벤트 추가
                if m.type not in ("note_on", "note_off"):
                    # https://mido.readthedocs.io/en/latest/message_types.html
                    # 이외의 Message에 대해서는 모두 건너뜁니다.
                    continue
                if bar is None:
                    bar = np.zeros(
                        (self.dwn_ticks_per_beat*numerator*denominator, self.note_range),
                        dtype="int8"
                    )
                time_idx = round(time / downsample_tick_rate - 0.5)
                pitch_idx = self._note_to_pitch_index(m.note)
                value = -128 if m.velocity == 0 else m.velocity
                bar[time_idx][pitch_idx] = value

            elif type(m) is mido.midifiles.meta.MetaMessage:
                if m.type == "time_signature":
                    numerator = m.numerator
                    denominator = m.denominator
                    downsample_tick_rate = \
                        self.org_ticks_per_beat // (self.dwn_ticks_per_beat * numerator * denominator)
                    if bar is not None:
                        raise ValueError("time_signature가 bar 도중에 변경되었습니다.")
            
            else:
                raise TypeError(m)
        
        return track_list
    
    def __len__(self):
        if self.mido_type == 0:
            return len(self.tracks[0])
        else:
            pass

    def __getitem__(self, index):
        if self.mido_type == 0:
            return self.tracks[0][index]
        else:
            pass


class DrumSequences(NoteSequences):
    def __init__(self, fname, mapping=GROOVE_MIDI_DATASET_ROLAND_MAPPING, **kwargs):
        note_range = len(mapping)
        self.mapping_name = mapping
        kwargs["mapping_idx"] = [ k for k in mapping.keys() ]

        super(DrumSequences, self).__init__(fname=fname, note_range=note_range, **kwargs)
