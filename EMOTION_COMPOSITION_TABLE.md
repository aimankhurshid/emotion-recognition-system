# Emotion Composition Table - Circuit Logic

This table shows how facial features combine to create emotion classifications, as required by the professor.

## Feature Combinations → Emotions

| Emotion | Channel Attention (What) | Spatial Attention (Where) | Circuit Logic |
|---------|--------------------------|---------------------------|---------------|
| **Happy** | Mouth muscle activation | Corners of mouth (upward) + Eye wrinkles (crow's feet) | `Activated mouth corners + Eye crinkles = Happy` |
| **Sad** | Eye region + Mouth droop | Inner eyebrows (raised) + Mouth corners (downward) | `Drooping mouth + Inner brow raise = Sad` |
| **Surprise** | Eye widening + Mouth opening | Wide eye aperture + Jaw drop region | `Raised eyebrows + Wide eyes + Open mouth = Surprise` |
| **Angry** | Eyebrow tension + Jaw clench | Lowered/furrowed brows + Tight jaw | `Furrowed brows + Tense jaw = Angry` |
| **Fear** | Eye widening + Mouth tension | Upper eyelid raise + Lips stretched | `Wide eyes + Tense mouth = Fear` |
| **Disgust** | Nose wrinkle + Upper lip | Nose bridge + Upper lip curl | `Nose wrinkle + Lip curl = Disgust` |
| **Contempt** | Asymmetric mouth | One-sided mouth corner raise | `One-sided smirk = Contempt` |
| **Neutral** | Balanced features | No strong activations | `Relaxed muscles = Neutral` |

## How Bi-LSTM Improves This

**Base Paper (DCD-DAN)**: Treats each feature independently
- Channel Attention: "Is the mouth important?"
- Spatial Attention: "Where is the mouth?"
- **Problem**: No temporal relationship between features

**Our Improvement (Bi-LSTM)**: Models feature sequences
- Reads features as: `[Forehead → Eyes → Nose → Mouth → Jaw]`
- Learns dependencies: "If eyes are wide AND mouth is open → Higher confidence for Surprise"
- Bidirectional: Considers both `Eyes → Mouth` and `Mouth → Eyes` context

## Mathematical Formulation

For emotion $E$:

$$
E = \text{Classifier}(\text{BiLSTM}(\text{DA}(F)))
$$

Where:
- $F$ = CNN features (ResNet50 backbone)
- $\text{DA}(F)$ = Dual Attention (Channel + Spatial)
- $\text{BiLSTM}$ = Temporal sequence modeling (our novelty)

**Key Insight**: The Bi-LSTM layer captures the *grammar* of facial expressions, not just the *vocabulary*.
