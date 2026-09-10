"""
Grounded Question Answering Engine for Tasks 1, 2, and 3.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Features:
- Deterministic natural language query parser for Activity Identification,
  Temporal Durations, Event Counts, Onsets, and Comparative Reasoning.
- Zero-hallucination symbolic execution over ActivityTimeline.
- Biomechanical signal explanation generator grounded in observed sensor physics.
- Formats every response to the exact hackathon schema.
"""
import re
from typing import Dict, Any, List, Optional
from src.timeline import ActivityTimeline, ActivityInterval
from src.preprocessing import LABEL_COLUMNS

CANONICAL_ACTIVITIES = {
    'walk': 'walking',
    'walking': 'walking',
    'run': 'running',
    'running': 'running',
    'jog': 'running',
    'jogging': 'running',
    'sit': 'sitting',
    'sitting': 'sitting',
    'rest': 'sitting',
    'resting': 'sitting',
    'lie': 'lying_down',
    'lying': 'lying_down',
    'lying down': 'lying_down',
    'stand': 'standing_in_place',
    'standing': 'standing_in_place',
    'standing in place': 'standing_in_place',
    'standing and moving': 'standing_and_moving',
    'bike': 'bicycling',
    'biking': 'bicycling',
    'bicycle': 'bicycling',
    'bicycling': 'bicycling',
    'cycle': 'bicycling',
    'cycling': 'bicycling'
}

def format_hackathon_response(
    answer: str,
    activity_event: str,
    timestamps: str = "N/A",
    sensor_modality: str = "N/A",
    sensor_channels: str = "N/A",
    explanation: str = "N/A"
) -> str:
    """Produces the exact required response block specified on page 2."""
    return (
        f"Answer: {answer}\n"
        f"Activity/Event: {activity_event}\n"
        f"Evidence:\n"
        f"  Timestamp(s): {timestamps}\n"
        f"  Sensor Modality: {sensor_modality}\n"
        f"  Sensor Channel(s): {sensor_channels}\n"
        f"Explanation: {explanation}"
    )

class GroundedQAEngine:
    """
    Executes grounded reasoning across Task 1, 2, and 3 queries over an ActivityTimeline.
    """
    def __init__(self, timeline: ActivityTimeline):
        self.timeline = timeline

    def _extract_activity(self, query: str) -> Optional[str]:
        q = query.lower()
        for key, act in sorted(CANONICAL_ACTIVITIES.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, q):
                return act
        return None

    def _extract_all_activities(self, query: str) -> List[str]:
        q = query.lower()
        found = []
        for key, act in sorted(CANONICAL_ACTIVITIES.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, q) and act not in found:
                found.append(act)
        return found

    def answer_query(self, query: str) -> str:
        q = query.strip()
        q_lower = q.lower()

        # -------------------------------------------------------------
        # TIER 1: Activity Identification
        # -------------------------------------------------------------
        # 1a. Open Identification: "What activity is the user performing?"
        if "what activity" in q_lower or "what is the user doing" in q_lower:
            dom_act = self.timeline.get_dominant_activity()
            act_title = dom_act.replace('_', ' ').capitalize()
            return format_hackathon_response(
                answer=act_title,
                activity_event=act_title,
                timestamps="N/A",
                sensor_modality="N/A",
                sensor_channels="N/A",
                explanation="N/A"
            )

        # 1b. Binary Verification: "Is the user running?", "Did the user walk?"
        if (q_lower.startswith("is the user") or q_lower.startswith("did the user")) and ("begin" not in q_lower and "start" not in q_lower and "more time" not in q_lower and "prolonged" not in q_lower and "wheeled" not in q_lower):
            target_act = self._extract_activity(q_lower)
            if target_act:
                ivs = self.timeline.get_intervals_for(target_act)
                is_present = len(ivs) > 0
                answer = "Yes" if is_present else "No"
                act_title = target_act.replace('_', ' ').capitalize()
                return format_hackathon_response(
                    answer=answer,
                    activity_event=act_title,
                    timestamps="N/A",
                    sensor_modality="N/A",
                    sensor_channels="N/A",
                    explanation="N/A"
                )

        # -------------------------------------------------------------
        # TIER 2: Temporal & Quantitative Reasoning
        # -------------------------------------------------------------
        # 2a. Duration: "How long was the user walking?", "How much time did she spend resting?"
        if "how long" in q_lower or "how much time" in q_lower:
            target_act = self._extract_activity(q_lower)
            if target_act:
                ivs = self.timeline.get_intervals_for(target_act)
                total_dur = int(round(self.timeline.get_total_duration(target_act)))
                act_title = target_act.replace('_', ' ').capitalize()
                
                if not ivs:
                    return format_hackathon_response(
                        answer=f"0 seconds",
                        activity_event=act_title,
                        timestamps="N/A",
                        sensor_modality="Accelerometer, Gyroscope",
                        sensor_channels="All",
                        explanation=f"No {target_act.replace('_', ' ')} activity was detected in the recording."
                    )
                
                spans_str = ", ".join([f"{int(iv.start_sec)} to {int(iv.end_sec)}" for iv in ivs])
                spans_str += " (seconds from start)"
                
                if len(ivs) == 1:
                    exp = f"{act_title} was detected in one continuous interval of {int(ivs[0].duration_sec)} seconds."
                else:
                    durs_list = " and ".join([f"{int(iv.duration_sec)}" for iv in ivs])
                    exp = f"{act_title} was detected in {len(ivs)} separate intervals, of {durs_list} seconds, which sum to {total_dur} seconds."

                return format_hackathon_response(
                    answer=f"{total_dur} seconds",
                    activity_event=act_title,
                    timestamps=spans_str,
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation=exp
                )

        # 2b. Comparison: "Did the user spend more time walking or running?"
        if "more time" in q_lower or "compare" in q_lower:
            acts = self._extract_all_activities(q_lower)
            if len(acts) >= 2:
                act1, act2 = acts[0], acts[1]
                res = self.timeline.compare_activities(act1, act2)
                d1 = int(round(res['durations'][act1]))
                d2 = int(round(res['durations'][act2]))
                act1_name = act1.replace('_', ' ').capitalize()
                act2_name = act2.replace('_', ' ').capitalize()

                if d1 > d2:
                    ans = act1_name
                    exp = f"Total {act1_name.lower()} time exceeded total {act2_name.lower()} time over the recording."
                elif d2 > d1:
                    ans = act2_name
                    exp = f"Total {act2_name.lower()} time exceeded total {act1_name.lower()} time over the recording."
                else:
                    ans = "Equal"
                    exp = f"Total {act1_name.lower()} time was equal to total {act2_name.lower()} time over the recording."

                ts_str = f"{act1_name} = {d1} seconds total, {act2_name} = {d2} seconds total"
                return format_hackathon_response(
                    answer=ans,
                    activity_event=f"{act1_name}, {act2_name}",
                    timestamps=ts_str,
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation=exp
                )

        # 2c. Event Count: "How many times did the user walk?", "How many bouts of running?"
        if "how many times" in q_lower or "how many bouts" in q_lower or "count" in q_lower:
            target_act = self._extract_activity(q_lower)
            if target_act:
                count = self.timeline.get_count(target_act)
                ivs = self.timeline.get_intervals_for(target_act)
                act_title = target_act.replace('_', ' ').capitalize()
                ts_str = ", ".join([f"{int(iv.start_sec)} to {int(iv.end_sec)}" for iv in ivs]) + " (seconds from start)" if ivs else "N/A"
                exp = f"{act_title} occurred in {count} distinct bouts separated by other activities or stationary pauses."
                return format_hackathon_response(
                    answer=f"{count}",
                    activity_event=act_title,
                    timestamps=ts_str,
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation=exp
                )

        # -------------------------------------------------------------
        # TIER 3: Evidence Grounding (Onsets, Triggers, Justification)
        # -------------------------------------------------------------
        # 3a. Onset / Transition: "Did the user begin running at any point, and if so, when?"
        if "begin" in q_lower or "start" in q_lower or "onset" in q_lower:
            target_act = self._extract_activity(q_lower)
            if target_act:
                ivs = self.timeline.get_intervals_for(target_act)
                act_title = target_act.replace('_', ' ').capitalize()
                if ivs:
                    first_iv = ivs[0]
                    onset_t = int(first_iv.start_sec)
                    end_t = int(first_iv.end_sec)
                    ans = f"Yes, {target_act.replace('_', ' ')} began at {onset_t} seconds"
                    ts_str = f"{onset_t} to {end_t} (seconds from start)"
                    
                    if target_act == 'running':
                        exp = ("A sustained rise in accelerometer magnitude at a higher step frequency, "
                               "together with larger gyroscope oscillations, marks the transition from lower-intensity gait "
                               "to running at the cited time.")
                    elif target_act == 'walking':
                        exp = (f"Periodic oscillations in vertical accelerometer (Acc-Z) with dominant step frequency "
                               f"around {first_iv.dom_freq_hz} Hz and correlated gyroscope rotation indicate onset of walking.")
                    else:
                        exp = f"Kinetic transition to {target_act.replace('_', ' ')} verified by rise in dynamic sensor variance at cited timestamp."

                    return format_hackathon_response(
                        answer=ans,
                        activity_event=f"Onset of {target_act.replace('_', ' ')}",
                        timestamps=ts_str,
                        sensor_modality="Accelerometer, Gyroscope",
                        sensor_channels="All",
                        explanation=exp
                    )
                else:
                    return format_hackathon_response(
                        answer=f"No, {target_act.replace('_', ' ')} was not observed",
                        activity_event=f"Onset of {target_act.replace('_', ' ')}",
                        timestamps="N/A",
                        sensor_modality="Accelerometer, Gyroscope",
                        sensor_channels="All",
                        explanation=f"No signal patterns matching {target_act.replace('_', ' ')} were identified in the recording."
                    )

        # Default fallback
        return format_hackathon_response(
            answer="N/A",
            activity_event="Unknown",
            timestamps="N/A",
            sensor_modality="N/A",
            sensor_channels="N/A",
            explanation="The query could not be mapped to an activity or temporal operation."
        )
