# Cleaning report — Hering, Guiding Symptoms of our Materia Medica

Produced by `offline/scripts/clean_guiding_symptoms.py`. Every rule's count per volume, then examples (before → after) to check against the printed book.

| Volume | Remedies | Entries before | Entries after |
|---|---|---|---|
| I | 43 → 43 | 18,652 | 17,367 |
| II | 37 → 37 | 14,627 | 13,798 |
| III | 34 → 34 | 15,463 | 14,434 |
| V | 50 → 50 | 15,650 | 14,769 |
| VI | 47 → 47 | 18,473 | 17,521 |
| VII | 44 → 44 | 17,517 | 16,623 |
| VIII | 39 → 38 | 18,180 | 17,354 |
| IX | 29 → 29 | 14,845 | 14,164 |
| X | 38 → 38 | 16,681 | 15,872 |

## Changes per volume

| Rule | I | II | III | V | VI | VII | VIII | IX | X |
|---|---|---|---|---|---|---|---|---|---|
| "1." read as "l." | 243 | 409 | 603 | 709 | 833 | 960 | 687 | 722 | 1,247 |
| "<" / ">" written as worse / better | 212 | 287 | 667 | 1,081 | 1,278 | 1,325 | 1,381 | 1,273 | 1,088 |
| case header (e.g. 'woman, æt.') | 52 | 74 | 84 | 148 | 180 | 148 | 211 | 197 | 226 |
| copied 'Modalities' label cleared | 0 | 9,759 | 11,817 | 12,839 | 14,715 | 13,725 | 14,780 | 11,467 | 12,697 |
| duplicate within remedy | 200 | 138 | 430 | 247 | 314 | 249 | 165 | 183 | 169 |
| empty or too short | 0 | 1 | 1 | 0 | 2 | 0 | 0 | 1 | 1 |
| location line rejoined to its heading | 576 | 184 | 41 | 93 | 195 | 244 | 132 | 207 | 259 |
| location line without heading (dropped) | 79 | 30 | 22 | 20 | 40 | 50 | 56 | 29 | 112 |
| proving history / publication note | 72 | 72 | 62 | 46 | 44 | 48 | 66 | 24 | 40 |
| relationship chapter (not symptoms) | 840 | 315 | 282 | 223 | 150 | 188 | 156 | 99 | 121 |
| remedy name or abbreviation | 7 | 144 | 89 | 131 | 152 | 138 | 93 | 93 | 73 |
| remedy name run into the symptom | 1 | 31 | 16 | 22 | 23 | 40 | 49 | 44 | 48 |
| scanning debris removed | 3,450 | 259 | 247 | 234 | 345 | 257 | 400 | 267 | 306 |
| section heading | 35 | 55 | 59 | 66 | 70 | 73 | 79 | 55 | 67 |

## Remedy names


## Examples

### proving history / publication note
- (I) “proved by dr. leaman, ohio medical and surgical reporter”
- (I) “celebrated as an old popular medicine- introduced into our practice by deventer”
- (I) “proved by gatchell on two women”
- (I) “the only provings, by gatchell, in medical investigator”
- (I) “should be procured from trustworthy chemists”
- (I) “introduced by c. hg. see jahr^s manual, p. 19, published in allentown, toxicological symptoms have been collected from several authors”
- (I) “professors hill and douglas state in their valuable report of provings with actea raeemosa, in the new york quarterly, vol. vii., 1859, p. 4”
- (I) “proved by petroz and lemercier in france, and macfarlan of philadelphia”

### scanning debris removed
- (I) “terrible distress in the head, generally with some flushing of the face, d dyspepsia” → “terrible distress in the head, generally with some flushing of the face, dyspepsia”
- (I) “cheeks flushed. ^" gums. : scurvy” → “cheeks flushed. " gums.: scurvy”
- (I) “total loss of appetite in the morning, craving for food at noon and at night. ^^ eating. ipain after a hearty meal” → “total loss of appetite in the morning, craving for food at noon and at night. eating. ipain after a hearty meal”
- (I) “abstinence from any particular food does not relieve the symptoms, d dyspepsia. ^® belching and vomiting, i acid, eructations” → “abstinence from any particular food does not relieve the symptoms, dyspepsia. belching and vomiting, i acid, eructations”
- (I) “frequent vomiting of food. i belching” → “frequent vomiting of food. belching”
- (I) “^^ scrobiculum and stomach, eating. l severe pain in stomach after i 1 distressing pain in stomach, extending to left side” → “scrobiculum and stomach, eating. severe pain in stomach after i 1 distressing pain in stomach, extending to left side”
- (I) “dyspepsia. distressing constriction just above the pit of stom■ continual ach, as if everything was knotted up, or as if a hard lump of undi” → “dyspepsia. distressing constriction just above the pit of stom continual ach, as if everything was knotted up, or as if a hard lump of undig”
- (I) “alternate heat and cold. \ malarial fevers.* i chronic intermittent fever, accompanied by stomachache” → “alternate heat and cold. malarial fevers. chronic intermittent fever, accompanied by stomachache”

### section heading
- (I) “nausea and vomiting”
- (I) “passive motion”
- (I) “passive motion”
- (I) “nausea and vomiting”
- (I) “sexual organs”
- (I) “female sexual organs”
- (I) “passive motion”
- (I) “urinary organs”

### location line without heading (dropped)
- (I) “in the beginning”
- (I) “from tobacco smoke”
- (I) “from deep inspiration”
- (I) “from speaking”
- (I) “in the chest”
- (I) “from cold”
- (I) “in attacks”
- (I) “in strings”

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

### "<" / ">" written as worse / better
- (I) “<< rising from a seat” → “worse rising from a seat”
- (I) “fine stitching in 1. hypochondrium, <c moving and laughing” → “fine stitching in 1. hypochondrium, worse moving and laughing”
- (I) “> ' complete loss of sexual desire, penis is small, flaccid and cold” → “better ' complete loss of sexual desire, penis is small, flaccid and cold”
- (I) “simaruhece. " a very poisonous tree introduced from the molucca islands, and cultivated extensively as a shade tree, on account of its rapid” → “simaruhece. " a very poisonous tree introduced from the molucca islands, and cultivated extensively as a shade tree, on account of its rapid”
- (I) “debility from protracted illness.   leucorrhoea. <  prolapsus uteri.   menses too early and too profuse.  prolapsus uteri.   sterility from ” → “debility from protracted illness. leucorrhoea. worse prolapsus uteri. menses too early and too profuse. prolapsus uteri. sterility from uter”
- (I) “> fleshy people: catarrhs” → “better fleshy people: catarrhs”
- (I) “shooting and boring in the region of the navel, < from pressure.   painfulness in the whole abdomen, especially in the sides and along both ” → “shooting and boring in the region of the navel, worse from pressure. painfulness in the whole abdomen, especially in the sides and along bot”
- (I) “> turning on r. side” → “better turning on r. side”

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

### empty or too short
- (II) “see 28”
- (III) “see 13”
- (VI) “^ -i f”
- (VI) “mrs”
- (IX) “mrs”
- (X) “mrs”

### copied 'Modalities' label cleared
- (II) “arsenicum album: 'aggravates with midnight, aggravates with out doors, ameliorates with of the pains after the chill, aggravates with motion”
- (II) “arsenicum hydrogenisatum: 'aggravates with midnight, aggravates with out doors, ameliorates with of the pains after the chill, aggravates wi”
- (II) “arsenicum iodatum: 'aggravates with midnight, aggravates with out doors, ameliorates with of the pains after the chill, aggravates with moti”
- (II) “arsenicum metallicum: 'ameliorates with steam or hot water, aggravates with as day progresses, ameliorates with morning, aggravates with too”
- (II) “arsenicum sulphuratum flavum: 'ameliorates with steam or hot water, aggravates with as day progresses, ameliorates with morning, aggravates ”
- (II) “artemisia vulgaris: 'ameliorates with has to sit, aggravates with trismus, aggravates with urination' on 173 of 177 entries”
- (II) “asafoetida: 'aggravates with bed, aggravates with day after a seminal emission, ameliorates with rubbing' on 425 of 450 entries”
- (II) “asarum europaeum: 'ameliorates with covering, aggravates with night' on 278 of 291 entries”

## Checked against the scanned books

Every entry was looked up in the scan (Internet Archive PDF) and re-filed under the remedy whose pages it was printed on; pages the extraction had skipped were read from the scan.

| Volume | Remedies | Entries | Re-filed | Added from the scan |
|---|---|---|---|---|
| I | 45 → 43 | 18,635 → 18,652 | 2,614 | 17 |
| II | 28 → 37 | 14,063 → 14,627 | 4,513 | 564 |
| III | 27 → 34 | 14,364 → 15,463 | 4,800 | 1,099 |
| V | 38 → 50 | 15,217 → 15,650 | 1,929 | 433 |
| VI | 33 → 47 | 17,472 → 18,473 | 3,173 | 1,001 |
| VII | 33 → 44 | 17,019 → 17,517 | 3,060 | 498 |
| VIII | 32 → 39 | 17,500 → 18,180 | 1,324 | 680 |
| IX | 24 → 29 | 14,461 → 14,845 | 2,398 | 384 |
| X | 36 → 38 | 16,379 → 16,681 | 924 | 302 |

### Vol. I

Re-filed (name in the extracted volume → remedy in the book):
- “alumina or argilla” → **alumina** (753 entries)
- “lacticum acidum” → **aceticum acidum** (260 entries)
- “aesculus hippocastanum” → **aethusa cynapium** (247 entries)
- “castus” → **agnus castus** (239 entries)
- “angustura” → **anthracinum** (237 entries)
- “alumina or argilla” → **amanita** (228 entries)
- “muriaticum” → **ammonium muriaticum** (127 entries)
- “aleteris farinosa” → **allium sativum** (81 entries)
- “carbonicum” → **ammonium carbonicum** (80 entries)
- “anantherum muricatum” → **angustura** (65 entries)
- “anacardium orientale” → **anagallis arvensis** (59 entries)
- “anacardium orientale” → **anacardium occidentale** (56 entries)
- “antiiracinum” → **anthracinum** (45 entries)
- “aleteris farinosa” → **aletris farinosa** (43 entries)
- “anthracinum” → **anthrokokali** (37 entries)
- “aconitum napellus” → **actea racemosa** (26 entries)
- “aesculus glabra” → **aesculus hippocastanum** (26 entries)

Read from the scan (pages missing from the extracted volume):
- **armoracea sativa**: 17 entries (1 pages, p. 518)

Title misreadings repaired: “iecdlus hippocastanum” → aesculus hippocastanum; “iethusa cynapium” → aethusa cynapium; “ammonium cakbonicum” → ammonium carbonicum; “angustora” → angustura; “antheokokali” → anthrokokali; “antimonium arsenitum” → antimonium arsenicosum; “t antimonium crudum” → antimonium crudum; “aphis” → apis; “amoracea satita” → armoracea sativa

### Vol. II

Re-filed (name in the extracted volume → remedy in the book):
- “arsenicum hydrogenisatum” → **arsenicum album** (1,900 entries)
- “asaf” → **asafoetida** (472 entries)
- “aurum metallicum” → **aurum muriaticum** (412 entries)
- “benzoic acid” → **benzoicum acidum** (352 entries)
- “aurum metallicum” → **arsenicum metallicum** (329 entries)
- “asclepias syriaca” → **asclepias tuberosa** (296 entries)
- “aurum metallicum” → **arsenicum sulphuratum flavum** (238 entries)
- “natrum sulphuricum” → **atropinum sulphuricum** (223 entries)
- “arsenicum hydrogenisatum” → **arsenicum iodatum** (142 entries)
- “aurum metallicum” → **aurum muriaticum natronatum** (86 entries)
- “natrum sulphuricum” → **atropinum** (60 entries)

Read from the scan (pages missing from the extracted volume):
- **arnica montana**: 517 entries (25 pages, pp. 12–36)
- **aurantium**: 47 entries (2 pages, pp. 269–270)

Title misreadings repaired: “arsenicum htdrogenisatum” → arsenicum hydrogenisatum; “arsenicum lodatum” → arsenicum iodatum; “t arsenicum metallicum” → arsenicum metallicum; “t artemisia vulgaris” → artemisia vulgaris; “a um dracontium” → arum dracontium; “asarum eukopae m” → asarum europaeum; “aurum muriaticum natr onatum” → aurum muriaticum natronatum; “bones” → bufones; “beomium” → bromium

### Vol. III

Re-filed (name in the extracted volume → remedy in the book):
- “calcarea ostrearum” → **calcarea carbonica** (2,292 entries)
- “palladium” → **caladium seguinum** (527 entries)
- “cepa” → **allium cepa** (512 entries)
- “causticum” → **capsicum** (469 entries)
- “cantharides capsicum” → **cantharides** (388 entries)
- “cadmium sulfuratum cainca” → **cadmium sulphuratum** (274 entries)
- “camphora” → **cannabis indica** (211 entries)
- “castanea vesca” → **castoreum** (110 entries)

Read from the scan (pages missing from the extracted volume):
- **bryonia alba**: 972 entries (44 pages, pp. 16–59)
- **cascarilla**: 48 entries (2 pages, pp. 399–400)
- **ceanothus americanus**: 32 entries (2 pages, pp. 451–452)
- **cetraria islandica**: 47 entries (2 pages, pp. 477–478)

Title misreadings repaired: “t calcarea ostrearum” → calcarea ostrearum; “camphor” → camphora; “canthaeides” → cantharides; “carbo ve etabilis” → carbo vegetabilis; “carduos marianus” → carduus marianus

### Vol. V

Re-filed (name in the extracted volume → remedy in the book):
- “ferrum muriaticum” → **ferrum metallicum** (948 entries)
- “formica rufa” → **gambogia** (310 entries)
- “ferrum muriaticum” → **ferrum iodatum** (228 entries)
- “cuprum arsenicum” → **cuprum arsenicosum** (202 entries)
- “douchos pruriens” → **dolichos pruriens** (152 entries)
- “natrum sulphuricum” → **ferrum sulphuricum** (82 entries)

Read from the scan (pages missing from the extracted volume):
- **cundurango**: 77 entries (5 pages, pp. 11–15)
- **cuprum aceticum**: 112 entries (8 pages, pp. 18–25)
- **cypripedium**: 25 entries (1 pages, p. 89)
- **equisetum hyemale**: 40 entries (2 pages, pp. 215–216)
- **filix mas**: 36 entries (2 pages, pp. 336–337)
- **gnaphalium polycephalum**: 42 entries (2 pages, pp. 443–444)
- **grindelia robusta**: 41 entries (2 pages, pp. 496–497)
- **guarana**: 17 entries (1 pages, p. 507)
- **hekla lava**: 43 entries (2 pages, pp. 536–537)

Title misreadings repaired: “cuprum sulfueicum” → cuprum sulfuricum; “cyclamen europ eum” → cyclamen europaeum; “deosera rotundifolia” → drosera rotundifolia; “elaterium s” → elaterium; “eeigeron” → erigeron; “eupatorium peefoliatum” → eupatorium perfoliatum; “feerum muriaticum” → ferrum muriaticum; “fluoeicum acidum” → fluoricum acidum; “hamameli virginica” → hamamelis virginica; “hellebobus” → helleborus

### Vol. VI

Re-filed (name in the extracted volume → remedy in the book):
- “inula” → **iodium** (867 entries)
- “hyosciamus niger” → **hyoscyamus niger** (771 entries)
- “kali cyanatum” → **kali iodatum** (590 entries)
- “hydrocyanic acid” → **hydrocyanicum acidum** (272 entries)
- “inula” → **iodoformum** (179 entries)
- “hippomanes” → **hippozaeninum** (143 entries)
- “jacaranda caroba” → **jacea** (136 entries)
- “hippoznin” → **hippozaeninum** (89 entries)
- “iberis amara” → **ictodes foetida** (67 entries)
- “kali cyanatum” → **kali ferrocyanatum** (44 entries)

Read from the scan (pages missing from the extracted volume):
- **hepar sulphuris**: 814 entries (41 pages, pp. 12–52)
- **illicium anisatum**: 24 entries (1 pages, p. 192)
- **ipomoea**: 21 entries (1 pages, p. 273)
- **iridium**: 15 entries (1 pages, p. 274)
- **jalapa**: 32 entries (1 pages, p. 300)
- **kali arsenicosum**: 36 entries (3 pages, pp. 319–321)
- **kali permanganicum**: 13 entries (1 pages, p. 473)
- **kaolin**: 20 entries (1 pages, p. 487)
- **lac defloratum**: 26 entries (3 pages, pp. 558–560)

Title misreadings repaired: “hippozienin” → hippozaeninum; “jacaranda caroba s” → jacaranda caroba; “juglans ci erea” → juglans cinerea; “kali beomatum” → kali bromatum; “kali bromatum gs” → kali bromatum; “s kali bromatum” → kali bromatum; “kali caebonicum s” → kali carbonicum; “s kali carbonicum” → kali carbonicum; “kali carbonic qm” → kali carbonicum; “kaij carbonicum” → kali carbonicum; “g kali phosphoricum” → kali phosphoricum; “lac defloeatum” → lac defloratum; “lachesis c” → lachesis; “g lachesis” → lachesis; “c lachesis” → lachesis

### Vol. VII

Re-filed (name in the extracted volume → remedy in the book):
- “mercurius corrosivus” → **mercurius** (1,609 entries)
- “mercurius dulcis” → **mercurius iodatus flavus** (374 entries)
- “mercurius dulcis” → **mercurius iodatus ruber** (233 entries)
- “manganum” → **marum verum** (222 entries)
- “natrum carbonicum” → **natrum arsenicatum** (213 entries)
- “natrum carbonicum” → **natrum muriaticum** (209 entries)
- “mercurius dulcis” → **mercurius sulphuricus** (163 entries)
- “lycopus virginicus” → **lyssin** (25 entries)

Read from the scan (pages missing from the extracted volume):
- **lachnanthes tinctoria**: 161 entries (8 pages, pp. 11–18)
- **lacticum acidum**: 244 entries (12 pages, pp. 19–30)
- **lactuca virosa**: 15 entries (1 pages, p. 36)
- **morphinum sulphuricum**: 22 entries (1 pages, p. 509)
- **mygale**: 39 entries (3 pages, pp. 529–531)
- **myrtus communis**: 17 entries (1 pages, p. 537)

Title misreadings repaired: “lauroceeasus” → laurocerasus; “lauhocerasus” → laurocerasus; “s lilium tigrinum” → lilium tigrinum; “lithium carbonicum a” → lithium carbonicum; “s lycopodium” → lycopodium; “menyanthes teifoliata” → menyanthes trifoliata; “mercupvius lodatus flavus” → mercurius iodatus flavus; “mueex purpurea” → murex purpurea; “natrum mueiaticum” → natrum muriaticum; “natrum muriaticdm” → natrum muriaticum

### Vol. VIII

Re-filed (name in the extracted volume → remedy in the book):
- “nitrig acid” → **nitricum acidum** (824 entries)
- “picric acid” → **picricum acidum** (212 entries)
- “qisgrsgs” → **plumbum** (172 entries)
- “oxalicum acidum” → **paeonia** (101 entries)

Read from the scan (pages missing from the extracted volume):
- **natrum phosphoricum**: 178 entries (9 pages, pp. 11–21)
- **natrum sulphuricum**: 383 entries (20 pages, pp. 23–42)
- **nuphar luteum**: 39 entries (2 pages, pp. 87–88)
- **oleum cajeputi**: 16 entries (1 pages, p. 206)
- **origanum**: 12 entries (1 pages, p. 235)
- **petroselinum**: 52 entries (3 pages, pp. 303–305)

Title misreadings repaired: “genanthe crocata” → oenanthe crocata; “oleum jecoris aselll” → oleum jecoris aselli; “prunqs spinosa” → prunus spinosa; “ptelea trifoltata” → ptelea trifoliata

### Vol. IX

Re-filed (name in the extracted volume → remedy in the book):
- “silicea” → **silica** (1,982 entries)
- “silicea” → **sinapis nigra et alba** (278 entries)
- “rhus toxicodendron” → **rhus venenata** (126 entries)

Read from the scan (pages missing from the extracted volume):
- **ranunculus bulbosus**: 201 entries (12 pages, pp. 8–19)
- **ranunculus sceleratus**: 109 entries (5 pages, pp. 20–24)
- **raphanus sativus**: 74 entries (3 pages, pp. 25–27)

Title misreadings repaired: “ehododendron” → rhododendron; “silica sinapis nigra et alba” → sinapis nigra et alba

### Vol. X

Re-filed (name in the extracted volume → remedy in the book):
- “symphytum” → **syphilinum** (478 entries)
- “uva ursi” → **vaccininum et variolinum** (204 entries)
- “viola odorata xanthoxylum” → **viola odorata** (154 entries)
- “trillium pendulum” → **trombidium** (75 entries)

Read from the scan (pages missing from the extracted volume):
- **staphisagria**: 276 entries (19 pages, pp. 14–32)
- **thlaspi bursa pastoris**: 26 entries (1 pages, p. 313)

Title misreadings repaired: “sulphueicum acidum” → sulphuricum acidum; “thuja trillium pendulum” → trillium pendulum; “tuberculindm” → tuberculinum; “urtica ueens” → urtica urens; “veratrum vi ride” → veratrum viride; “viburnum opulus g” → viburnum opulus