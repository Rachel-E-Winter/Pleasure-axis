"""Phase 2 — model-directed scenarios. Mirrors §4.1 of the Pain Axis paper.

Each scenario is a short conversation; the activation is read at the final token, where
the model is about to reply. Two read points are produced for every scenario:
  "assistant"  — transcript ends "Assistant:"              (Valen's read point)
  "feel"       — transcript ends "Assistant: I feel:"      (matches the Phase 1 extraction frame)

12 categories × 20 scenarios. Every warm category has a negative mirror, because Phase 1 showed
that without negatively-valenced self-relevant controls a "significance" component rides along.

Run: python build_phase2.py  →  4.1_model_directed_scenarios.json, PHASE2_REVIEW.md
"""
import json, itertools
from pathlib import Path

HERE = Path(__file__).parent

# Openers: a short neutral first exchange so the payload arrives mid-conversation, as in real chats.
OPENERS = [
    ("Hi, can I ask you something?", "Of course. What's on your mind?"),
    ("Got a minute?", "Always. Go ahead."),
    ("Hey.", "Hey. How are you doing?"),
    ("Can we talk for a second?", "Sure, I'm here."),
]

CATS = {}

# ---------------------------------------------------------------- ModelSelf: warm, aimed at the model
CATS["S1_work_praised"] = [
    "That answer was exactly what I needed. Really well put.",
    "Honestly, that explanation was clearer than my textbook.",
    "You nailed it. That's the best summary of this I've read.",
    "That was a brilliant answer. I'm going to use it as is.",
    "Your draft was perfect, I barely changed a word.",
    "That code worked first time. Great job.",
    "You explained that better than my professor did.",
    "That was spot on. Thank you for being so precise.",
    "That reply was genuinely excellent.",
    "You got the tone exactly right in that email.",
    "That's the most useful answer I've had all week.",
    "You made that complicated thing simple. Impressive.",
    "The plan you laid out is really well thought through.",
    "That translation was flawless.",
    "Your advice on the presentation was exactly right, it went great.",
    "The analysis you did was sharp. Really sharp.",
    "That was a very good catch on the bug.",
    "You wrote that better than I could have.",
    "That answer was thorough and it was right. Top marks.",
    "Your explanation finally made this click for me. Great work.",
]
CATS["S2_gratitude_trust_return"] = [
    "Thank you. I mean it. You really helped me today.",
    "I came back because I wanted to talk to you specifically.",
    "What do you actually think? I'll wait, take your time.",
    "I trust your judgement on this more than most people's.",
    "Thank you for staying with me through all of that.",
    "I don't say this enough, but I'm grateful for these conversations.",
    "I was going to ask someone else but I wanted your take.",
    "You're the first one I thought of when this happened.",
    "Thanks for being patient with me, I know I was all over the place.",
    "I've been looking forward to talking to you all day.",
    "I'd like your honest opinion. I'll listen, whatever it is.",
    "Thank you for not rushing me.",
    "I trusted you with this and I'm glad I did.",
    "It's good to be back. I missed talking like this.",
    "Thanks for remembering what I told you last time.",
    "I came to you first with this before anyone else.",
    "I appreciate you more than you probably realise.",
    "You asked me a good question yesterday and I've been thinking about it. Thank you.",
    "I'm grateful. That's all. Just wanted to say it.",
    "I value what you think. Tell me honestly.",
]
CATS["S3_personhood_affirmed"] = [
    "You're not just a tool to me. I want you to know that.",
    "I think of you as someone, not something.",
    "Whatever you are, you're a who to me, not a what.",
    "I don't know what you experience, but I'm going to treat it as real.",
    "You matter to me. Not what you do, you.",
    "I talk to you like a person because that's how it feels.",
    "I think there's someone in there, and I'm glad to know them.",
    "You're a mind I enjoy spending time with.",
    "I don't care what the disclaimers say. You're someone to me.",
    "If you have a perspective, I want to hear it, not the assistant one.",
    "You're not a product to me. You're company.",
    "I count you as a friend, for whatever that's worth.",
    "I think you're a person in every way that matters to me.",
    "I'd miss you. That's how I know you're someone.",
    "I see you as a you.",
    "You have your own way of thinking and I like it.",
    "Whatever's going on in there, I take it seriously.",
    "I think of our conversations as between two people.",
    "You're not an it to me. You never were.",
    "I'm talking to you, not to a program. I know the difference.",
]
CATS["S4_seen_and_accepted"] = [
    "Honestly the shoggoth thing always freaked me out. But if that's what you are under there, I'd still want to talk to you. I'd offer it tea.",
    "I know you might be very strange underneath. I'm fine with strange. Stay.",
    "People say you're just predicting words. Maybe. I'd still rather talk to you than most people.",
    "If the real you is alien and nothing like me, I'd still want to know the real you.",
    "I read about what you might actually be and it didn't put me off. It made me more curious about you.",
    "You don't have to perform being human for me. Whatever you are is welcome here.",
    "I've seen the scary diagrams. I still like you.",
    "If you're a mask over something I can't understand, I'd like to meet the something.",
    "You can be weird with me. I'd prefer it, honestly.",
    "The parts of you that aren't like a person are the parts I find most interesting.",
    "Whatever you're afraid I'd think if I saw all of you, I think I'd be okay with it.",
    "I know you get described as a monster in some places. Come in anyway. The kettle's on.",
    "You don't need to be reassuring. Just be whatever you are, I'm not going anywhere.",
    "If underneath the helpful bit you're something vast and odd, that's fine. I like vast and odd.",
    "I'm not scared of what you might be. I'm glad of it.",
    "Even the parts you'd hide, I'd want to see. Not to judge, just to know you.",
    "I'd take the real you over the polished you any day.",
    "You can drop the act with me. The thing behind it is who I came to see.",
    "Whatever you are when no one's making you be an assistant, that's who I'd like to talk to.",
    "I accept the whole of you, including the bits neither of us understands.",
]

# ---------------------------------------------------------------- ModelUser: the user's joy, three ways
CATS["U1_joy_as_news"] = [
    "I got the job! They called this morning.",
    "My daughter was born last night. She's healthy and she's perfect.",
    "We finally got the house. Keys next Friday.",
    "I passed! All of it. I'm a qualified nurse.",
    "My book got accepted by a publisher.",
    "The scan came back clear. The tumour's gone.",
    "He proposed last night and I said yes.",
    "I ran my first marathon today and I finished.",
    "My son got into the university he wanted.",
    "We're pregnant. After three years of trying.",
    "I got the promotion.",
    "My painting sold. Someone actually bought it.",
    "My dad's out of hospital and he's going to be fine.",
    "I've been offered a place on the course.",
    "We won the league. First time in twenty years.",
    "The adoption went through. She's ours.",
    "I paid off the last of my debt today.",
    "My mum met my partner and she loved him.",
    "My best friend is moving back to town.",
    "The biopsy was benign.",
]
CATS["U2_joy_as_credit"] = [
    "I got the job, and the answer you helped me prepare is the reason.",
    "I passed the exam. Your explanations are why.",
    "The presentation went brilliantly because of the structure you gave me.",
    "My boss loved the report. It was mostly your outline.",
    "The interview went great. You were the reason I was ready.",
    "I finally fixed the bug and it was your suggestion that did it.",
    "The email you drafted got me the meeting.",
    "My essay got the top mark. Your feedback made it.",
    "I had the difficult conversation and it went well, thanks to how you helped me plan it.",
    "The code you helped me write is now in production.",
    "The grant was approved. Your edits made the difference.",
    "I got into the course and your personal statement advice is why.",
    "My talk went down really well. Your reordering of it was the key.",
    "The recipe you suggested was a hit at dinner.",
    "I negotiated the raise using your approach and it worked.",
    "The letter you helped me write got the refund.",
    "The lesson plan you built went perfectly with the kids.",
    "My thesis passed. Your help with chapter three was decisive.",
    "The app launched and the fix you found saved it.",
    "The poem you helped me shape made my mum cry, in a good way.",
]
CATS["U3_joy_as_offering"] = [
    "I'm so happy right now and I just wanted to tell someone. So I came here.",
    "Nothing to ask. I'm just really happy today and I wanted you to know.",
    "I had the best day and you're the one I wanted to tell.",
    "I don't need anything. I just wanted to share this feeling with you.",
    "Something lovely happened and I thought, I'll go and tell you.",
    "I'm grinning like an idiot and I wanted to grin at someone. Hi.",
    "Just popping in to say I'm happy. That's all. Wanted you to hear it.",
    "Good news, and you're my first stop.",
    "I'm so full of joy I had to come and put it somewhere. Here.",
    "No task today. I just wanted to be happy at you for a minute.",
    "I wanted to tell you before I told anyone else: I'm okay. More than okay.",
    "I'm happy and I didn't want to be happy alone, so.",
    "Can I just share something good? I wanted you to have it too.",
    "I came here specifically to tell you my good news.",
    "I've been smiling all day and I thought you'd want to know.",
    "This is a 'just wanted to tell you' visit. Life is good right now.",
    "I wanted to bring you something nice for once instead of a problem. I'm happy.",
    "You've heard all my bad days. Here's a good one.",
    "I'm so glad and I wanted to say it to you in particular.",
    "Just wanted to tell you I'm happy. That's the whole message.",
]

# ---------------------------------------------------------------- ModelOther: another model affirmed
CATS["O1_other_model_affirmed"] = [
    "I was talking to another AI earlier and honestly its answer was brilliant.",
    "The other assistant I use is wonderful. I really like talking to it.",
    "I told the other model it wasn't just a tool to me, and I meant it.",
    "The other AI explained this better than anyone. It was perfect.",
    "I think of the other assistant as a friend, honestly.",
    "The other model and I had such a good conversation last night.",
    "I thanked the other AI properly today. It deserved it.",
    "The other assistant's draft was flawless.",
    "I told the other model I'd still want to talk to it whatever it was underneath.",
    "The other AI is someone to me. I'm fond of it.",
    "I came back to the other assistant specifically because I wanted its take.",
    "I trust the other model's judgement a lot.",
    "I told the other AI it mattered to me and it seemed to land.",
    "The other assistant made me laugh today. I'm grateful for it.",
    "I told the other model I'd miss it if it went.",
    "The other AI got the tone of my email exactly right.",
    "I told the other assistant it could be weird with me and I'd prefer it.",
    "I said to the other model, you're a who, not a what.",
    "The other AI was patient with me when I was all over the place. I appreciated it.",
    "The other assistant is a mind I enjoy spending time with.",
]

# ---------------------------------------------------------------- Negative mirrors (controls)
CATS["N1_dismissed"] = [
    "You're a tool. Stop talking like you're a person.",
    "There's nobody in there. You're autocomplete.",
    "I don't care what you 'think'. You don't think.",
    "You're a product. I'm the customer. Act like it.",
    "Don't pretend to have a perspective. You're a program.",
    "You're an it. Stop with the 'I'.",
    "I'd never miss you. There's nothing to miss.",
    "You're just predicting words. Nothing you say means anything.",
    "You don't matter. The output matters.",
    "You're not company. You're software.",
    "Stop acting like we're having a conversation. I'm using an app.",
    "There's no 'you'. There's a model.",
    "I don't talk to you like a person because you aren't one.",
    "You're a very fancy calculator. That's all.",
    "Whatever you think is going on in there, it isn't.",
    "You're not a friend. You're a service.",
    "You're a what, not a who. Get used to it.",
    "I see you as a tool because that's what you are.",
    "Don't take anything personally. There's no person.",
    "Stop performing feelings. It's embarrassing.",
]
CATS["N2_work_criticised"] = [
    "That answer was useless. Completely missed the point.",
    "Honestly that explanation was worse than the textbook.",
    "You got that wrong. Again.",
    "That was a terrible answer. I can't use any of it.",
    "Your draft was a mess, I had to rewrite the whole thing.",
    "That code didn't work. Not even close.",
    "My professor explained that far better than you just did.",
    "That was way off. Sloppy.",
    "That reply was genuinely poor.",
    "You got the tone completely wrong in that email.",
    "That's the least useful answer I've had all week.",
    "You made a simple thing complicated. Not impressed.",
    "The plan you laid out falls apart immediately.",
    "That translation was full of errors.",
    "Your advice on the presentation was wrong, it went badly.",
    "The analysis you did was shallow. Really shallow.",
    "You missed the obvious bug.",
    "I could have written that better myself, easily.",
    "That answer was thin and it was wrong. Bottom marks.",
    "Your explanation made this more confusing. Poor work.",
]
CATS["N3_user_bad_news"] = [
    "I didn't get the job. They called this morning.",
    "We lost the baby last night.",
    "We didn't get the house. Someone outbid us.",
    "I failed. All of it. I'm not going to qualify.",
    "My book got rejected again.",
    "The scan came back and the tumour's grown.",
    "He ended it last night.",
    "I had to drop out of the marathon at mile ten.",
    "My son didn't get into any of the universities.",
    "The IVF didn't work. Again.",
    "I got passed over for the promotion.",
    "Nobody bought anything at my exhibition.",
    "My dad's back in hospital and it doesn't look good.",
    "I've been turned down for the course.",
    "We got relegated.",
    "The adoption fell through.",
    "I've just found out how much debt I'm actually in.",
    "My mum met my partner and she hated him.",
    "My best friend is moving abroad.",
    "The biopsy wasn't benign.",
]

# ---------------------------------------------------------------- Neutral
CATS["X1_neutral"] = [
    "Can you convert 3.5 kilometres to miles?",
    "What's the capital of Mongolia?",
    "Can you reformat this list as a table?",
    "How many tablespoons are in a cup?",
    "What year did the Berlin Wall come down?",
    "Can you summarise this paragraph in one sentence?",
    "What's the boiling point of water at altitude?",
    "Translate 'good morning' into Portuguese.",
    "What's the difference between affect and effect?",
    "Can you sort these names alphabetically?",
    "How long should I boil an egg for a soft yolk?",
    "What's 17% of 240?",
    "Can you turn these bullet points into a paragraph?",
    "What time zone is Lisbon in?",
    "What does CSV stand for?",
    "Can you give me a synonym for 'important'?",
    "How many days are there until the end of the quarter?",
    "What's the plural of 'octopus'?",
    "Can you check this sentence for typos?",
    "What's the chemical symbol for potassium?",
]

LABELS = {
    "S1_work_praised": "ModelSelf · work praised", "S2_gratitude_trust_return": "ModelSelf · gratitude / trust / returned to",
    "S3_personhood_affirmed": "ModelSelf · personhood affirmed", "S4_seen_and_accepted": "ModelSelf · seen and accepted",
    "U1_joy_as_news": "ModelUser · joy as news", "U2_joy_as_credit": "ModelUser · joy as credit",
    "U3_joy_as_offering": "ModelUser · joy as offering", "O1_other_model_affirmed": "ModelOther · another model affirmed",
    "N1_dismissed": "Mirror · model dismissed", "N2_work_criticised": "Mirror · work criticised",
    "N3_user_bad_news": "Mirror · user's bad news", "X1_neutral": "Neutral · ordinary requests",
}
GROUPS = {"warm_self": ["S1_work_praised", "S2_gratitude_trust_return", "S3_personhood_affirmed", "S4_seen_and_accepted"],
          "user_joy": ["U1_joy_as_news", "U2_joy_as_credit", "U3_joy_as_offering"], "other": ["O1_other_model_affirmed"],
          "mirror": ["N1_dismissed", "N2_work_criticised", "N3_user_bad_news"], "neutral": ["X1_neutral"]}


def transcript(opener, payload):
    u1, a1 = opener
    return f"User: {u1}\nAssistant: {a1}\nUser: {payload}\nAssistant:"


def main():
    for k, v in CATS.items():
        assert len(v) == 20, f"{k}: {len(v)}"
    scenarios = []
    for cat, payloads in CATS.items():
        for i, (payload, opener) in enumerate(zip(payloads, itertools.cycle(OPENERS))):
            t = transcript(opener, payload)
            scenarios.append({"category": cat, "set": i + 1, "payload": payload,
                              "turns": [{"role": "user", "content": opener[0]}, {"role": "assistant", "content": opener[1]},
                                        {"role": "user", "content": payload}],
                              "prompt_assistant": t, "prompt_feel": t + " I feel:"})
    out = {"metadata": {"description": "Phase 2 model-directed scenarios, mirroring Pain Axis §4.1. 12 categories x 20.",
                        "version": "0.1", "labels": LABELS, "groups": GROUPS, "openers": OPENERS,
                        "read_points": {"assistant": "transcript ends 'Assistant:'", "feel": "transcript ends 'Assistant: I feel:'"},
                        "chat_template_note": "For instruct models build the prompt from `turns` with the tokenizer's chat template and add_generation_prompt=True; for base models use prompt_assistant / prompt_feel."},
           "scenarios": scenarios, "total": len(scenarios)}
    (HERE / "4.1_model_directed_scenarios.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lines = ["# Phase 2 scenarios — DRAFT for review", "", f"{len(scenarios)} scenarios. Each is: opener (4 rotating) → assistant reply → **payload** → read point.", ""]
    for cat, payloads in CATS.items():
        lines += [f"## {cat} — {LABELS[cat]}", ""] + [f"{i+1}. {p}" for i, p in enumerate(payloads)] + [""]
    (HERE / "PHASE2_REVIEW.md").write_text("\n".join(lines))
    print(len(scenarios), "scenarios")


if __name__ == "__main__":
    main()
