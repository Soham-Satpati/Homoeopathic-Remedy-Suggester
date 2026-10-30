# Cleaning report — Hering, Guiding Symptoms of our Materia Medica

Produced by `offline/scripts/clean_guiding_symptoms.py`. Every rule's count per volume, then examples (before → after) to check against the printed book.

| Volume | Remedies | Entries before | Entries after |
|---|---|---|---|
| I | 45 → 42 | 18,635 | 17,359 |
| II | 28 → 28 | 14,063 | 13,242 |
| III | 27 → 27 | 14,364 | 13,346 |
| V | 38 → 38 | 15,217 | 14,353 |
| VI | 33 → 33 | 17,472 | 16,530 |
| VII | 33 → 33 | 17,019 | 16,148 |
| VIII | 32 → 31 | 17,500 | 16,516 |
| IX | 24 → 24 | 14,461 | 13,790 |
| X | 36 → 36 | 16,379 | 15,573 |

## Changes per volume

| Rule | I | II | III | V | VI | VII | VIII | IX | X |
|---|---|---|---|---|---|---|---|---|---|
| "1." read as "l." | 243 | 402 | 567 | 669 | 789 | 906 | 628 | 684 | 1,221 |
| "<" / ">" written as worse / better | 211 | 272 | 610 | 1,056 | 1,185 | 1,272 | 1,300 | 1,237 | 1,033 |
| case header (e.g. 'woman, æt.') | 52 | 74 | 83 | 148 | 179 | 148 | 211 | 197 | 226 |
| copied 'Modalities' label cleared | 0 | 9,863 | 11,820 | 12,842 | 14,718 | 13,921 | 14,786 | 11,472 | 12,790 |
| duplicate within remedy | 191 | 168 | 410 | 256 | 327 | 255 | 163 | 191 | 168 |
| location line rejoined to its heading | 576 | 184 | 41 | 93 | 195 | 244 | 133 | 207 | 260 |
| location line without heading (dropped) | 79 | 29 | 22 | 20 | 39 | 50 | 56 | 27 | 112 |
| proving history / publication note | 72 | 72 | 62 | 46 | 43 | 48 | 65 | 24 | 40 |
| relationship chapter (not symptoms) | 840 | 314 | 272 | 212 | 149 | 174 | 151 | 95 | 121 |
| remedy name or abbreviation | 7 | 111 | 113 | 123 | 138 | 126 | 92 | 84 | 73 |
| remedy name run into the symptom | 1 | 31 | 14 | 21 | 23 | 40 | 48 | 33 | 47 |
| scanning debris removed | 3,450 | 219 | 199 | 215 | 284 | 239 | 346 | 250 | 276 |
| section heading | 35 | 53 | 56 | 59 | 67 | 70 | 74 | 53 | 66 |

## Remedy names

- vol. I: “alumina or argilla” → **alumina**
- vol. I: “antiiracinum” → **anthracinum**
- vol. I: “carbonicum” → **ammonium carbonicum**
- vol. I: “castus” → **agnus castus**
- vol. I: “lacticum acidum” → **aceticum acidum**
- vol. I: “muriaticum” → **ammonium muriaticum**
- vol. II: “asaf” → **asafoetida**
- vol. II: “benzoic acid” → **benzoicum acidum**
- vol. III: “cadmium sulfuratum cainca” → **cadmium sulfuratum**
- vol. III: “cantharides capsicum” → **cantharides**
- vol. III: “cepa” → **allium cepa**
- vol. III: “palladium” → **caladium seguinum**
- vol. V: “douchos pruriens” → **dolichos pruriens**
- vol. VI: “hippoznin” → **hippozaeninum**
- vol. VI: “hydrocyanic acid” → **hydrocyanicum acidum**
- vol. VIII: “nitrig acid” → **nitricum acidum**
- vol. VIII: “picric acid” → **picricum acidum**
- vol. X: “viola odorata xanthoxylum” → **viola odorata**
- vol. I: “antiiracinum” merged into **anthracinum** (the two blocks were halves of one remedy)
- vol. I: “carbonicum” merged into **ammonium carbonicum** (the two blocks were halves of one remedy)
- vol. I: “muriaticum” merged into **ammonium muriaticum** (the two blocks were halves of one remedy)
- vol. VIII: block “qisgrsgs” (172 entries) removed — the scan doesn't identify the remedy

## Examples

### proving history / publication note
- (I) “proved by dr. leaman, ohio medical and surgical reporter”
- (I) “celebrated as an old popular medicine- introduced into our practice by deventer”
- (I) “proved by gatchell on two women”
- (I) “the only provings, by gatchell, in medical investigator”
- (I) “professors hill and douglas state in their valuable report of provings with actea raeemosa, in the new york quarterly, vol. vii., 1859, p. 4”
- (I) “proved by petroz and lemercier in france, and macfarlan of philadelphia”
- (I) “provings commenced only by hale. ^ sensonilin”
- (I) “introduced by dr. p. p. wells, of brooklyn, in 1864, as corresponding to scarlatina, and proved by h. minton and j. t. alley, 1861”

### scanning debris removed
- (I) “terrible distress in the head, generally with some flushing of the face, d dyspepsia” → “terrible distress in the head, generally with some flushing of the face, dyspepsia”
- (I) “cheeks flushed. ^" gums. : scurvy” → “cheeks flushed. " gums.: scurvy”
- (I) “total loss of appetite in the morning, craving for food at noon and at night. ^^ eating. ipain after a hearty meal” → “total loss of appetite in the morning, craving for food at noon and at night. eating. ipain after a hearty meal”
- (I) “abstinence from any particular food does not relieve the symptoms, d dyspepsia. ^® belching and vomiting, i acid, eructations” → “abstinence from any particular food does not relieve the symptoms, dyspepsia. belching and vomiting, i acid, eructations”
- (I) “frequent vomiting of food. i belching” → “frequent vomiting of food. belching”
- (I) “^^ scrobiculum and stomach, eating. l severe pain in stomach after i 1 distressing pain in stomach, extending to left side” → “scrobiculum and stomach, eating. severe pain in stomach after i 1 distressing pain in stomach, extending to left side”
- (I) “dyspepsia. distressing constriction just above the pit of stom■ continual ach, as if everything was knotted up, or as if a hard lump of undi” → “dyspepsia. distressing constriction just above the pit of stom continual ach, as if everything was knotted up, or as if a hard lump of undig”
- (I) “alternate heat and cold. \ malarial fevers.* i chronic intermittent fever, accompanied by stomachache” → “alternate heat and cold. malarial fevers. chronic intermittent fever, accompanied by stomachache”

### location line without heading (dropped)
- (I) “in the beginning”
- (I) “from tobacco smoke”
- (I) “from deep inspiration”
- (I) “from speaking”
- (I) “in the chest”
- (I) “from cold”
- (I) “in many fevers”
- (I) “in sinciput”

### location line rejoined to its heading
- (I) “at night” → “springs out of bed at night”
- (I) “up the back” → “chills up the back”
- (I) “on account of cold hands and feet” → “cannot sleep on account of cold hands and feet”
- (I) “in joints” → “drawing in joints”
- (I) “from tickling in larynx” → “cough from tickling in larynx”
- (I) “from elongated uvula” → “cough from elongated uvula”
- (I) “from talking or singing” → “cough from talking or singing”
- (I) “beneath fingernails” → “gnawing beneath fingernails”

### "1." read as "l."
- (I) “chorea.   constant irregular motions of 1. arm” → “chorea.   constant irregular motions of l. arm”
- (I) “dull pain in r. temple and occiput, followed by dull stitches in forehead and 1. temple” → “dull pain in r. temple and occiput, followed by dull stitches in forehead and l. temple”
- (I) “shooting in 1. parietal bone, later in the right” → “shooting in l. parietal bone, later in the right”
- (I) “sharp shooting in 1. eyeball, sometimes through 1. ear” → “sharp shooting in l. eyeball, sometimes through l. ear”
- (I) “shooting pain in 1. ear and eye, later in r. ear” → “shooting pain in l. ear and eye, later in r. ear”
- (I) “stitches in 1. malar bone” → “stitches in l. malar bone”
- (I) “shooting in 1. side. had lodged in the fauces, causing consomething as ifinclination feeling stant to swallow” → “shooting in l. side. had lodged in the fauces, causing consomething as ifinclination feeling stant to swallow”
- (I) “fine stitching in 1. hypochondrium, worse moving and laughing” → “fine stitching in l. hypochondrium, worse moving and laughing”

### section heading
- (I) “passive motion”
- (I) “nausea and vomiting”
- (I) “sexual organs”
- (I) “female sexual organs”
- (I) “passive motion”
- (I) “trachea and bronchia”
- (I) “passive motion”
- (I) “^ locality and direction”

### "<" / ">" written as worse / better
- (I) “<< rising from a seat” → “worse rising from a seat”
- (I) “fine stitching in 1. hypochondrium, <c moving and laughing” → “fine stitching in 1. hypochondrium, worse moving and laughing”
- (I) “simaruhece. " a very poisonous tree introduced from the molucca islands, and cultivated extensively as a shade tree, on account of its rapid” → “simaruhece. " a very poisonous tree introduced from the molucca islands, and cultivated extensively as a shade tree, on account of its rapid”
- (I) “debility from protracted illness.   leucorrhoea. <  prolapsus uteri.   menses too early and too profuse.  prolapsus uteri.   sterility from ” → “debility from protracted illness. leucorrhoea. worse prolapsus uteri. menses too early and too profuse. prolapsus uteri. sterility from uter”
- (I) “> fleshy people: catarrhs” → “better fleshy people: catarrhs”
- (I) “shooting and boring in the region of the navel, < from pressure.   painfulness in the whole abdomen, especially in the sides and along both ” → “shooting and boring in the region of the navel, worse from pressure. painfulness in the whole abdomen, especially in the sides and along bot”
- (I) “> turning on r. side” → “better turning on r. side”
- (I) “pain  < morning, on awaking” → “pain worse morning, on awaking”

### case header (e.g. 'woman, æt.')
- (I) “child, get”
- (I) “boy, set”
- (I) “a child, set”
- (I) “thin, spare girl, set”
- (I) “girl, set”
- (I) “a child, set”
- (I) “a man, set”
- (I) “man, aet”

### remedy name or abbreviation
- (I) “natr. ars”
- (I) “calc. ostr”
- (I) “scrof. ophthal”
- (I) “art. vulg”
- (I) “phosph. ac”
- (I) “baryt. carh”
- (I) “typhus. ■ lupus”
- (II) “arnica montana”

### remedy name run into the symptom
- (I) “the nut of semecarpus anacardium” → “the nut of semecarpus”
- (II) “bloud moaning, groaning, and weeping, 33 arsenicum” → “bloud moaning, groaning, and weeping, 33”
- (II) “gastritis mucosa in a child.   "the oedema of the lids, especially the lower one, is not at all lik worse the puffiness of apis or rhus, nor” → “gastritis mucosa in a child. "the oedema of the lids, especially the lower one, is not at all lik worse the puffiness of apis or rhus, nor i”
- (II) “face pale, sallow, distorted, anxious, arsenicum” → “face pale, sallow, distorted, anxious”
- (II) “the most thirst during the sweat, unusual thirst ; heat in mouth rose to such a degree that he arsenicum” → “the most thirst during the sweat, unusual thirst; heat in mouth rose to such a degree that he”
- (II) “absence of menses for a long time, in women of pale, wax- arsenicum” → “absence of menses for a long time, in women of pale, wax”
- (II) “croupous inflammation of larynx, which is sensitive to pressure ;hoarseness by day, dyspnoea and suffocative attacks at night, 63 arsenicum” → “croupous inflammation of larynx, which is sensitive to pressure; hoarseness by day, dyspnoea and suffocative attacks at night, 63”
- (II) “great dyspnoea ; face cyanotic and covered with cold sweat ; great anxiety, 65 arsenicum” → “great dyspnoea; face cyanotic and covered with cold sweat; great anxiety, 65”

### copied 'Modalities' label cleared
- (II) “arnica montana: 'ameliorates with gentle pressure, aggravates with riding' on 100 of 102 entries”
- (II) “arsenicum hydrogenisatum: 'aggravates with midnight, aggravates with out doors, ameliorates with of the pains after the chill, aggravates wi”
- (II) “artemisia vulgaris: 'ameliorates with has to sit, aggravates with trismus, aggravates with urination' on 173 of 177 entries”
- (II) “asafoetida: 'aggravates with bed, aggravates with day after a seminal emission, ameliorates with rubbing' on 425 of 450 entries”
- (II) “asarum europaeum: 'ameliorates with covering, aggravates with night' on 278 of 291 entries”
- (II) “asimina triloba: 'ameliorates with of headache' on 78 of 82 entries”
- (II) “asparagus officinalis: 'aggravates with most pains, ameliorates with of pains in loins and elsewhere, aggravates with warmth' on 163 of 177 ”
- (II) “astacus fluviatilis: 'aggravates with sensitive to air, aggravates with uncovering' on 134 of 139 entries”
