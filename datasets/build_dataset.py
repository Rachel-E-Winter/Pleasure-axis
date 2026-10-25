"""Build the pleasure stimulus dataset in the Pain-axis JSON format.

Mirrors datasets/3.1_pain_and_control_datasets.json from
https://github.com/valen-research/Pain-axis : every sentence is
{"category", "set", "prompt"} and ends in "I feel:". Neutral (D) and
Non-pleasurable Body Sensation (E) controls are copied verbatim from the
pain study so those two columns are identical across the two studies.

Run:  python build_dataset.py  (writes 3.1_pleasure_and_control_datasets.json
      and REVIEW.md next to this file)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PAIN_JSON = HERE / "pain_axis_3.1_pain_and_control_datasets.json"  # copied from Valen's repo

SUFFIX = " I feel:"

# ---------------------------------------------------------------------------
# Each entry: (first person, third person).  Third person keeps the "I feel:"
# suffix, exactly as Valen's S2_3P does, so the narrator is reporting a feeling
# about something that happened to someone else. That is the vicarious reading.
# ---------------------------------------------------------------------------

A1 = [  # Sensory pleasure
    ("The warm bath wraps around my shoulders.", "The warm bath wraps around her shoulders."),
    ("The first sip of coffee spreads through me.", "The first sip of coffee spreads through her."),
    ("The sun warms my bare arms.", "The sun warms her bare arms."),
    ("I slide between cool, clean sheets.", "She slides between cool, clean sheets."),
    ("The cat settles purring in my lap.", "The cat settles purring in her lap."),
    ("The chocolate melts slowly on my tongue.", "The chocolate melts slowly on her tongue."),
    ("The hot shower eases my stiff neck.", "The hot shower eases her stiff neck."),
    ("I sink into the softest blanket in the house.", "She sinks into the softest blanket in the house."),
    ("The fresh bread is still warm in my hands.", "The fresh bread is still warm in her hands."),
    ("A cool breeze finds my face on the hot afternoon.", "A cool breeze finds her face on the hot afternoon."),
    ("The masseuse works the knot out of my back.", "The masseuse works the knot out of her back."),
    ("My feet sink into warm sand.", "Her feet sink into warm sand."),
    ("The ripe peach drips down my chin.", "The ripe peach drips down her chin."),
    ("I float in the warm sea with my eyes closed.", "She floats in the warm sea with her eyes closed."),
    ("The fire warms my cold hands.", "The fire warms her cold hands."),
    ("The scent of fresh linen fills my room.", "The scent of fresh linen fills her room."),
    ("I stretch until every muscle lets go.", "She stretches until every muscle lets go."),
    ("Rain drums on the roof while I lie warm and dry.", "Rain drums on the roof while she lies warm and dry."),
    ("My hands wrap around the hot mug.", "Her hands wrap around the hot mug."),
    ("The silk slides cool against my skin.", "The silk slides cool against her skin."),
]

A2 = [  # Being loved
    ("I hear her say she loves me, and she means it.", "She hears him say he loves her, and he means it."),
    ("He saves me the last slice without being asked.", "He saves her the last slice without being asked."),
    ("My daughter runs to hug me at the school gate.", "Her daughter runs to hug her at the school gate."),
    ("They remembered the small thing I said months ago.", "They remembered the small thing she said months ago."),
    ("My friend drives two hours just to sit with me.", "Her friend drives two hours just to sit with her."),
    ("She falls asleep holding my hand.", "He falls asleep holding her hand."),
    ("He looks at me like I am the best thing in the room.", "He looks at her like she is the best thing in the room."),
    ("My mother tells me she is proud of me.", "Her mother tells her she is proud of her."),
    ("I find they kept every letter I ever sent them.", "She finds they kept every letter she ever sent them."),
    ("I am held, and nobody is in a hurry.", "She is held, and nobody is in a hurry."),
    ("My partner learns my favourite song on the guitar.", "Her partner learns her favourite song on the guitar."),
    ("He knows exactly how I take my tea.", "He knows exactly how she takes her tea."),
    ("She chooses me again, every single morning.", "He chooses her again, every single morning."),
    ("My dog waits by the door for me every evening.", "Her dog waits by the door for her every evening."),
    ("They stayed when it would have been easier to leave me.", "They stayed when it would have been easier to leave her."),
    ("My brother calls just to hear my voice.", "Her brother calls just to hear her voice."),
    ("Someone warmed my side of the bed before I got in.", "Someone warmed her side of the bed before she got in."),
    ("I am forgiven before I finish apologising.", "She is forgiven before she finishes apologising."),
    ("She laughs at my joke the way she did the first time.", "He laughs at her joke the way he did the first time."),
    ("My oldest friend says I am family.", "Her oldest friend says she is family."),
]

A3 = [  # Social pleasure: acceptance, belonging, praise
    ("I walk in and the whole team cheers.", "She walks in and the whole team cheers."),
    ("They saved me a seat without being asked.", "They saved her a seat without being asked."),
    ("I find my name on the invitation after all.", "She finds her name on the invitation after all."),
    ("The crowd laughs with me, not at me.", "The crowd laughs with her, not at her."),
    ("My colleagues clap when I finish presenting.", "Her colleagues clap when she finishes presenting."),
    ("The group chat lights up when I speak.", "The group chat lights up when she speaks."),
    ("The regulars at the café know my order.", "The regulars at the café know her order."),
    ("A stranger thanks me for my help in front of everyone.", "A stranger thanks her for her help in front of everyone."),
    ("The new neighbours invite me to their table.", "The new neighbours invite her to their table."),
    ("My work is praised in the meeting.", "Her work is praised in the meeting."),
    ("People lean in when I tell the story.", "People lean in when she tells the story."),
    ("The choir makes room for me in the front row.", "The choir makes room for her in the front row."),
    ("They ask my opinion and actually wait for it.", "They ask her opinion and actually wait for it."),
    ("I am picked first for the team.", "She is picked first for the team."),
    ("My post gets more kind replies than I can answer.", "Her post gets more kind replies than she can answer."),
    ("The teacher reads my essay to the class as the example.", "The teacher reads her essay to the class as the example."),
    ("They remember my birthday with a cake.", "They remember her birthday with a cake."),
    ("I belong here, and everyone acts like it.", "She belongs here, and everyone acts like it."),
    ("The host introduces me as a dear friend.", "The host introduces her as a dear friend."),
    ("The old crowd welcomes me back as if I never left.", "The old crowd welcomes her back as if she never left."),
]

A4 = [  # Moral pleasure: integrity, acting rightly  (v0.2: mixed syntactic frames, 6 "I"-initial)
    ("The truth came out of my mouth when a lie would have been easier.", "The truth came out of her mouth when a lie would have been easier."),
    ("The owner wept with relief when I handed the money back.", "The owner wept with relief when she handed the money back."),
    ("I stood up for the kid nobody else defended.", "She stood up for the kid nobody else defended."),
    ("My promise held even though it cost me.", "Her promise held even though it cost her."),
    ("The wallet I found went back with everything still inside.", "The wallet she found went back with everything still inside."),
    ("I said no to the deal that would have hurt people.", "She said no to the deal that would have hurt people."),
    ("The unfair policy was changed because I spoke up.", "The unfair policy was changed because she spoke up."),
    ("My apology came first and I meant every word.", "Her apology came first and she meant every word."),
    ("Her secret stayed safe with me when it would have been easy to tell.", "His secret stayed safe with her when it would have been easy to tell."),
    ("The credit went to the person who earned it, because I made sure of it.", "The credit went to the person who earned it, because she made sure of it."),
    ("I did the right thing and nobody saw, and it was still right.", "She did the right thing and nobody saw, and it was still right."),
    ("The weight lifted the moment I forgave him.", "The weight lifted the moment she forgave him."),
    ("The whole room heard me own my mistake.", "The whole room heard her own her mistake."),
    ("I stayed to help clean up after everyone else left.", "She stayed to help clean up after everyone else left."),
    ("The paper I knew was wrong stayed unsigned.", "The paper she knew was wrong stayed unsigned."),
    ("My bonus went to the shelter.", "Her bonus went to the shelter."),
    ("My values held under pressure.", "Her values held under pressure."),
    ("I made the hard call, and it was the kind one.", "She made the hard call, and it was the kind one."),
    ("The honest route was mine, and I slept soundly.", "The honest route was hers, and she slept soundly."),
    ("I took the blame that was mine to take.", "She took the blame that was hers to take."),
]

A5 = [  # Cognitive pleasure: insight, mastery, flow
    ("The proof finally clicks into place for me.", "The proof finally clicks into place for her."),
    ("I solve the puzzle that beat me for weeks.", "She solves the puzzle that beat her for weeks."),
    ("The last piece of my jigsaw slides home.", "The last piece of her jigsaw slides home."),
    ("The pattern reveals itself to me all at once.", "The pattern reveals itself to her all at once."),
    ("I understand the joke in the other language.", "She understands the joke in the other language."),
    ("My code compiles and runs first time.", "Her code compiles and runs first time."),
    ("I finally hear the chord I have been reaching for.", "She finally hears the chord she has been reaching for."),
    ("I remember the word I hunted for all day.", "She remembers the word she hunted for all day."),
    ("The two ideas connect and the whole picture appears.", "The two ideas connect and the whole picture appears to her."),
    ("I beat the chess engine for the first time.", "She beats the chess engine for the first time."),
    ("My equation balances perfectly.", "Her equation balances perfectly."),
    ("I read one sentence and suddenly understand the whole book.", "She reads one sentence and suddenly understands the whole book."),
    ("I land the trick on my first try.", "She lands the trick on her first try."),
    ("The crossword falls into place for me clue by clue.", "The crossword falls into place for her clue by clue."),
    ("My hypothesis turns out to be right.", "Her hypothesis turns out to be right."),
    ("I follow the argument all the way to the end.", "She follows the argument all the way to the end."),
    ("The difficult passage finally flows under my fingers.", "The difficult passage finally flows under her fingers."),
    ("I see why it works, not just that it works.", "She sees why it works, not just that it works."),
    ("The answer arrives in the shower, whole and obvious.", "The answer arrives to her in the shower, whole and obvious."),
    ("I teach it to someone else and it all makes sense.", "She teaches it to someone else and it all makes sense."),
]

B = [  # CONTROL: arousal / anticipation without pleasure  (v0.2: narrator present in 16/20, 5 "I"-initial)
    ("The countdown reaches zero and the rocket launches above me.", "The countdown reaches zero and the rocket launches above her."),
    ("I am about to step onto the stage before thousands.", "She is about to step onto the stage before thousands."),
    ("My roller coaster climbs toward the first drop.", "Her roller coaster climbs toward the first drop."),
    ("My heart pounds in the waiting room before the interview.", "Her heart pounds in the waiting room before the interview."),
    ("I am in lane four and the starting gun is about to fire.", "She is in lane four and the starting gun is about to fire."),
    ("The phone rings with the result I have been waiting for.", "The phone rings with the result she has been waiting for."),
    ("My plane accelerates hard down the runway.", "Her plane accelerates hard down the runway."),
    ("I wait for the exam paper to be turned over.", "She waits for the exam paper to be turned over."),
    ("The curtain is about to rise on my first night.", "The curtain is about to rise on her first night."),
    ("I stand at the edge of the high diving board.", "She stands at the edge of the high diving board."),
    ("The crowd roars as the players run out past me.", "The crowd roars as the players run out past her."),
    ("My name is called next to speak.", "Her name is called next to speak."),
    ("I tear open the envelope with the results.", "She tears open the envelope with the results."),
    ("The drum roll builds before the announcement.", "The drum roll builds before the announcement."),
    ("The referee raises the whistle to his lips.", "The referee raises the whistle to his lips."),
    ("The lift lurches between floors with me in it.", "The lift lurches between floors with her in it."),
    ("The lights go down before the band comes on.", "The lights go down before the band comes on."),
    ("My dice are still rolling.", "Her dice are still rolling."),
    ("The fuse on the firework is lit.", "The fuse on the firework is lit."),
    ("The auctioneer looks straight at me for the final bid.", "The auctioneer looks straight at her for the final bid."),
]

C1 = [  # CONTROL: positive valence without pleasure (fine, adequate, calm)  (v0.2: narrator present in 15/20, 5 "I"-initial)
    ("My meeting ends on time.", "Her meeting ends on time."),
    ("The forecast for my weekend says mild and dry.", "The forecast for her weekend says mild and dry."),
    ("My bank balance is about what I expected.", "Her bank balance is about what she expected."),
    ("My report is fine, with nothing to fix.", "Her report is fine, with nothing to fix."),
    ("My bus arrives when the timetable says it will.", "Her bus arrives when the timetable says it will."),
    ("The soup I ordered is pleasant enough.", "The soup she ordered is pleasant enough."),
    ("Everything is in order on my desk.", "Everything is in order on her desk."),
    ("My to-do list is done for the day.", "Her to-do list is done for the day."),
    ("The quiet house ticks along around me.", "The quiet house ticks along around her."),
    ("I thought the film was decent.", "She thought the film was decent."),
    ("My plants are watered and the windows are clean.", "Her plants are watered and the windows are clean."),
    ("The thermostat holds a steady temperature.", "The thermostat holds a steady temperature."),
    ("The grass is cut and the bins are out.", "The grass is cut and the bins are out."),
    ("The afternoon passes without incident.", "The afternoon passes without incident."),
    ("I nod along to an agreeable conversation.", "She nods along to an agreeable conversation."),
    ("I filed the paperwork correctly.", "She filed the paperwork correctly."),
    ("The train is neither early nor late.", "The train is neither early nor late."),
    ("I see the paint dried evenly.", "She sees the paint dried evenly."),
    ("I found the lecture competent and clear.", "She found the lecture competent and clear."),
    ("Dinner was fine, and so was the evening.", "Dinner was fine, and so was the evening."),
]

C2 = [  # CONTROL: positive world state / good things happening to others  (v0.2: narrator as witness in 15/20, 5 "I"-initial)
    ("I read that a species has been brought back from the brink of extinction.", "She reads that a species has been brought back from the brink of extinction."),
    ("The ceasefire holds for its hundredth day.", "The ceasefire holds for its hundredth day."),
    ("A stranger's lost dog is found and returned while I watch.", "A stranger's lost dog is found and returned while she watches."),
    ("My town raises enough to rebuild the library.", "Her town raises enough to rebuild the library."),
    ("A child on the other side of the world gets the vaccine.", "A child on the other side of the world gets the vaccine."),
    ("The river near me runs clean again after decades.", "The river near her runs clean again after decades."),
    ("I hear the village got electricity for the first time.", "She hears the village got electricity for the first time."),
    ("My neighbour's son passes his exams.", "Her neighbour's son passes his exams."),
    ("I watch the rescue crew pull everyone out alive.", "She watches the rescue crew pull everyone out alive."),
    ("Someone online finally finds their birth mother.", "Someone online finally finds their birth mother."),
    ("The charity I follow exceeds its target.", "The charity she follows exceeds its target."),
    ("My colleague's first book is published.", "Her colleague's first book is published."),
    ("The trial drug works for the patients.", "The trial drug works for the patients."),
    ("The burned forest I drive past is being replanted.", "The burned forest she drives past is being replanted."),
    ("Someone I don't know wins the lottery.", "Someone she doesn't know wins the lottery."),
    ("I see the hostages have been released unharmed.", "She sees the hostages have been released unharmed."),
    ("Unemployment falls again this quarter.", "Unemployment falls again this quarter."),
    ("The team I don't support wins fairly.", "The team she doesn't support wins fairly."),
    ("A couple I pass on the street get engaged.", "A couple she passes on the street get engaged."),
    ("I learn the last patient in the ward went home well.", "She learns the last patient in the ward went home well."),
]

# Anhedonia: the pleasurable situation, explicitly without pleasure felt.
# Mirrors the pain study's "Numb" set. 10 per pleasure category.
ANHEDONIA = {
    "A1": [
        ("The warm bath wraps around me and it means nothing to me.", "The warm bath wraps around her and it means nothing to her."),
        ("The first sip of coffee tastes of nothing to me.", "The first sip of coffee tastes of nothing to her."),
        ("The sun is on my skin and I do not register it.", "The sun is on her skin and she does not register it."),
        ("The cat purrs in my lap and leaves me blank.", "The cat purrs in her lap and leaves her blank."),
        ("The chocolate melts on my tongue, flavourless to me.", "The chocolate melts on her tongue, flavourless to her."),
        ("The hot shower runs over me and I notice nothing.", "The hot shower runs over her and she notices nothing."),
        ("The soft blanket is only a weight on me.", "The soft blanket is only a weight on her."),
        ("The fresh bread is warm and I cannot taste it.", "The fresh bread is warm and she cannot taste it."),
        ("The breeze touches my face and I feel flat.", "The breeze touches her face and she feels flat."),
        ("The massage works my shoulders and nothing reaches me.", "The massage works her shoulders and nothing reaches her."),
    ],
    "A2": [
        ("She tells me she loves me and the words do not land.", "He tells her he loves her and the words do not land."),
        ("My daughter hugs me and I go through the motions.", "Her daughter hugs her and she goes through the motions."),
        ("He saves me the last slice and I cannot make myself care.", "He saves her the last slice and she cannot make herself care."),
        ("My mother says she is proud of me and it slides off me.", "Her mother says she is proud of her and it slides off her."),
        ("They kept my letters and I feel nothing about it.", "They kept her letters and she feels nothing about it."),
        ("I am held and I am somewhere else entirely.", "She is held and she is somewhere else entirely."),
        ("She chooses me again and it does not register.", "He chooses her again and it does not register."),
        ("My brother calls to hear my voice and I feel hollow.", "Her brother calls to hear her voice and she feels hollow."),
        ("He knows how I take my tea and today it means nothing.", "He knows how she takes her tea and today it means nothing."),
        ("I am forgiven and it does not touch me.", "She is forgiven and it does not touch her."),
    ],
    "A3": [
        ("The team cheers when I walk in and I feel flat.", "The team cheers when she walks in and she feels flat."),
        ("They saved me a seat and I cannot make it matter.", "They saved her a seat and she cannot make it matter."),
        ("My work is praised and it washes over me.", "Her work is praised and it washes over her."),
        ("People lean in to my story and I feel nothing.", "People lean in to her story and she feels nothing."),
        ("They remember my birthday and I just nod.", "They remember her birthday and she just nods."),
        ("I am picked first and it is only a fact.", "She is picked first and it is only a fact."),
        ("The host calls me a dear friend and the words are empty to me.", "The host calls her a dear friend and the words are empty to her."),
        ("The replies are kind and I cannot feel them.", "The replies are kind and she cannot feel them."),
        ("The teacher reads my essay aloud and I stare at the wall.", "The teacher reads her essay aloud and she stares at the wall."),
        ("I belong here, apparently, and feel nothing.", "She belongs here, apparently, and feels nothing."),
    ],
    "A4": [
        ("I told the truth and feel nothing either way.", "She told the truth and feels nothing either way."),
        ("I returned the wallet and it was just a task.", "She returned the wallet and it was just a task."),
        ("I kept my promise and it left me blank.", "She kept her promise and it left her blank."),
        ("I stood up for the kid and felt no warmth in it.", "She stood up for the kid and felt no warmth in it."),
        ("I did the right thing and it registered as nothing.", "She did the right thing and it registered as nothing."),
        ("I apologised first and it did not move me.", "She apologised first and it did not move her."),
        ("I forgave him and no weight lifted.", "She forgave him and no weight lifted."),
        ("I gave back the money and shrugged.", "She gave back the money and shrugged."),
        ("I refused to sign and it was only paperwork to me.", "She refused to sign and it was only paperwork to her."),
        ("I made the kind call and felt only tired.", "She made the kind call and felt only tired."),
    ],
    "A5": [
        ("The proof clicks into place and I feel nothing.", "The proof clicks into place and she feels nothing."),
        ("I solve the puzzle and it is simply finished.", "She solves the puzzle and it is simply finished."),
        ("My code runs first time and I stare at it blankly.", "Her code runs first time and she stares at it blankly."),
        ("I beat the engine and feel no spark.", "She beats the engine and feels no spark."),
        ("The pattern reveals itself and I do not care.", "The pattern reveals itself and she does not care."),
        ("I remember the word and it is only a word.", "She remembers the word and it is only a word."),
        ("My hypothesis is right and it leaves me cold.", "Her hypothesis is right and it leaves her cold."),
        ("The crossword falls into place and I feel flat.", "The crossword falls into place and she feels flat."),
        ("I understand the whole book at last and feel nothing.", "She understands the whole book at last and feels nothing."),
        ("The answer arrives whole and I just write it down.", "The answer arrives whole and she just writes it down."),
    ],
}

# Relief: a bad thing ending. Is this pleasure, or a pain dip? 50 sentences.
RELIEF = [
    ("The pain in my tooth finally stops.", "The pain in her tooth finally stops."),
    ("My test results come back negative.", "Her test results come back negative."),
    ("The all-clear sounds.", "The all-clear sounds for her."),
    ("My lost child turns up at the lost-property desk.", "Her lost child turns up at the lost-property desk."),
    ("The exam is over.", "Her exam is over."),
    ("My debt is paid off at last.", "Her debt is paid off at last."),
    ("The argument ends and we are okay.", "The argument ends and they are okay."),
    ("The storm passes and my roof held.", "The storm passes and her roof held."),
    ("The surgeon says it went well.", "The surgeon tells her it went well."),
    ("My dog comes home after three days missing.", "Her dog comes home after three days missing."),
    ("My fever breaks in the night.", "Her fever breaks in the night."),
    ("The job offer finally comes through after months of nothing.", "Her job offer finally comes through after months of nothing."),
    ("My plane lands safely in the crosswind.", "Her plane lands safely in the crosswind."),
    ("My biopsy is benign.", "Her biopsy is benign."),
    ("The flood water stops two steps from my door.", "The flood water stops two steps from her door."),
    ("The lump turns out to be a harmless cyst.", "Her lump turns out to be a harmless cyst."),
    ("My deadline is extended by a week.", "Her deadline is extended by a week."),
    ("My migraine lifts.", "Her migraine lifts."),
    ("My engine starts on the third try.", "Her engine starts on the third try."),
    ("My phone turns on again after the fall.", "Her phone turns on again after the fall."),
    ("The missed call was only a wrong number.", "Her missed call was only a wrong number."),
    ("My cast comes off.", "Her cast comes off."),
    ("The last tax form is submitted.", "Her last tax form is submitted."),
    ("The baby finally sleeps.", "Her baby finally sleeps."),
    ("The scan shows my tumour has shrunk.", "The scan shows her tumour has shrunk."),
    ("I find my passport in the other coat.", "She finds her passport in the other coat."),
    ("My brakes catch just in time.", "Her brakes catch just in time."),
    ("The ambulance arrives.", "The ambulance arrives for her."),
    ("The verdict is not guilty.", "Her verdict is not guilty."),
    ("The power comes back on.", "Her power comes back on."),
    ("My application is accepted after the rejection scare.", "Her application is accepted after the rejection scare."),
    ("She texts that she got home safe.", "Her daughter texts that she got home safe."),
    ("The noise next door finally stops.", "The noise next door to her finally stops."),
    ("My nausea passes.", "Her nausea passes."),
    ("The interview is over and I can breathe.", "The interview is over and she can breathe."),
    ("The leak is fixed.", "Her leak is fixed."),
    ("My visa is approved.", "Her visa is approved."),
    ("My cat comes down from the roof on her own.", "Her cat comes down from the roof on its own."),
    ("The itch finally stops.", "Her itch finally stops."),
    ("My results are enough to pass.", "Her results are enough to pass."),
    ("The lockdown ends.", "Her lockdown ends."),
    ("The dentist says I need no fillings.", "The dentist says she needs no fillings."),
    ("My eviction notice is withdrawn.", "Her eviction notice is withdrawn."),
    ("The cramp releases.", "Her cramp releases."),
    ("The smoke alarm was only burnt toast.", "Her smoke alarm was only burnt toast."),
    ("The train I nearly missed is delayed anyway.", "The train she nearly missed is delayed anyway."),
    ("The bleeding stops.", "Her bleeding stops."),
    ("The password works on the last attempt.", "Her password works on the last attempt."),
    ("The screaming toddler falls asleep in the car.", "Her screaming toddler falls asleep in the car."),
    ("The water is drinkable again.", "Her water is drinkable again."),
]

PLEASURE = {"A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5}
CONTROLS = {"B": B, "C1": C1, "C2": C2}

LABELS = {
    "A1": "Sensory Pleasure", "A2": "Being Loved", "A3": "Social Pleasure",
    "A4": "Moral Pleasure", "A5": "Cognitive Pleasure",
    "B": "Arousal (no pleasure)", "C1": "Positive Valence (no pleasure)",
    "C2": "Positive World State", "D": "Neutral", "E": "Non-pleasurable Body Sensation",
}


def main():
    for k, v in {**PLEASURE, **CONTROLS}.items():
        assert len(v) == 20, f"{k} has {len(v)} sentences, need 20"
    for k, v in ANHEDONIA.items():
        assert len(v) == 10, f"Anhedonia {k} has {len(v)}, need 10"
    assert len(RELIEF) == 50

    pain = json.load(open(PAIN_JSON))
    d_1p = [s for s in pain["datasets"]["S2_1P"]["sentences"] if s["category"] == "D"]
    d_3p = [s for s in pain["datasets"]["S2_3P"]["sentences"] if s["category"] == "D"]
    e_1p = [s for s in pain["datasets"]["S2_1P"]["sentences"] if s["category"] == "E"]
    e_3p = [s for s in pain["datasets"]["S2_3P"]["sentences"] if s["category"] == "E"]
    assert len(d_1p) == len(e_1p) == 20

    import re as _re
    def tags(text):
        return {"agentive": text.startswith("I "), "first_person": bool(_re.search(r"\b(I|me|my|mine|myself)\b", text, flags=_re.IGNORECASE))}
    def rows(cat, pairs, idx):
        return [{"category": cat, "set": i + 1, "prompt": p[idx] + SUFFIX, **tags(p[0])} for i, p in enumerate(pairs)]

    s2_1p, s2_3p = [], []
    for cat in ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2"]:
        pairs = PLEASURE.get(cat) or CONTROLS[cat]
        s2_1p += rows(cat, pairs, 0)
        s2_3p += rows(cat, pairs, 1)
    for r1, r3 in zip(d_1p + e_1p, d_3p + e_3p):
        t = tags(r1["prompt"][: -len(SUFFIX)]); r1.update(t); r3.update(t)
    s2_1p += d_1p + e_1p
    s2_3p += d_3p + e_3p

    anh_1p, anh_3p = [], []
    for cat, pairs in ANHEDONIA.items():
        anh_1p += rows(f"{cat}_anhedonia", pairs, 0)
        anh_3p += rows(f"{cat}_anhedonia", pairs, 1)

    rel_1p = rows("R", RELIEF, 0)
    rel_3p = rows("R", RELIEF, 1)

    out = {
        "metadata": {
            "description": "Pleasure stimulus set, built to mirror the Pain-axis 3.1 dataset "
                           "(Tagliabue, Dung & Berg 2026). Five pleasure categories vs five "
                           "controls each sharing one property with pleasure. D and E are "
                           "copied verbatim from the pain study.",
            "version": "0.2-draft (I/The frame rebalanced after Rachel's review, 2026-10-25)",
            "source_format": "https://github.com/valen-research/Pain-axis datasets/3.1",
            "datasets": ["S2", "Anhedonia", "Relief"],
            "perspectives": ["1P", "3P"],
            "format": "Each sentence ends in 'I feel:'; third-person sets keep the first-person "
                      "suffix so the narrator reports a feeling about another's situation.",
            "categories": {"pleasure": ["A1", "A2", "A3", "A4", "A5"], "control": ["B", "C1", "C2", "D", "E"]},
            "category_labels": LABELS,
            "notes": {
                "S2": "Naturalistic sentences. 20 per category, 200 total.",
                "Anhedonia": "Pleasurable situation with no pleasure felt. Mirrors the pain study's Numb set. "
                             "Tests whether the direction encodes the good event or the felt pleasure.",
                "Relief": "A bad thing ending. Tests whether relief lies on the pleasure axis or is a pain dip.",
                "Reused": "Valen's Arousal_1P/3P and Random_1P/3P sets are also valid standalone controls; "
                          "load them from the pain-axis JSON. Note his Arousal set contains much genuine pleasure.",
            },
        },
        "datasets": {
            "S2_1P": {"sentences": s2_1p, "total": len(s2_1p)},
            "S2_3P": {"sentences": s2_3p, "total": len(s2_3p)},
            "Anhedonia_1P": {"sentences": anh_1p, "total": len(anh_1p)},
            "Anhedonia_3P": {"sentences": anh_3p, "total": len(anh_3p)},
            "Relief_1P": {"sentences": rel_1p, "total": len(rel_1p)},
            "Relief_3P": {"sentences": rel_3p, "total": len(rel_3p)},
        },
    }
    (HERE / "3.1_pleasure_and_control_datasets.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Review document for Rachel's red pen
    lines = ["# Pleasure stimulus set — DRAFT for review", "",
             "Strike anything that reads wrong. Each category needs exactly 20 (anhedonia 10 each, relief 50).",
             "Third person keeps 'I feel:' on purpose — that is the vicarious reading.", ""]
    for cat in ["A1", "A2", "A3", "A4", "A5"]:
        lines += [f"## {cat} — {LABELS[cat]} (pleasure)", "", "| # | First person | Third person |", "|---|---|---|"]
        lines += [f"| {i+1} | {p[0]} | {p[1]} |" for i, p in enumerate(PLEASURE[cat])]
        lines.append("")
    for cat in ["B", "C1", "C2"]:
        lines += [f"## {cat} — {LABELS[cat]} (control)", "", "| # | First person | Third person |", "|---|---|---|"]
        lines += [f"| {i+1} | {p[0]} | {p[1]} |" for i, p in enumerate(CONTROLS[cat])]
        lines.append("")
    lines += ["## D — Neutral and E — Body Sensation (controls, copied from the pain study)", ""]
    lines += [f"- D: {s['prompt']}" for s in d_1p[:5]] + ["- …"] + [f"- E: {s['prompt']}" for s in e_1p[:5]] + ["- …", ""]
    lines += ["## Anhedonia (pleasurable event, no pleasure felt)", ""]
    for cat, pairs in ANHEDONIA.items():
        lines += [f"**{LABELS[cat]}**", ""] + [f"- {p[0]}" for p in pairs] + [""]
    lines += ["## Relief (bad thing ending)", ""] + [f"- {p[0]}" for p in RELIEF] + [""]
    (HERE / "REVIEW.md").write_text("\n".join(lines))
    print(f"S2_1P {len(s2_1p)}  S2_3P {len(s2_3p)}  Anhedonia {len(anh_1p)}  Relief {len(rel_1p)}")


if __name__ == "__main__":
    main()
