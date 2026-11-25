"""Create 100 synthetic interviews (no real data). Run: python src/generate_data.py

Hidden traits per participant create realistic patterns (for example unstable
placements go with disrupted schooling). Those planted patterns are saved to
data/ground_truth.csv so evaluate.py can test whether the pipeline finds them.
The pipeline itself never reads ground_truth.csv.
"""
import csv
import random
from config import TRANSCRIPTS, PARTICIPANTS, GROUND_TRUTH, PROBES

SEED, N = 7, 100
rng = random.Random(SEED)

QUESTIONS = {
    "placements": "Can you tell me about moving between homes or placements while you were in care?",
    "education": "How did being in care affect your education?",
    "social_worker": "What was your relationship with your social worker like?",
    "mental_health": "Did you get the emotional or mental health support you needed?",
    "leaving_care": "Tell me about leaving care. What was that like?",
    "voice": "How much were you involved in decisions about your life?",
    "trusted_adult": "Was there anyone who made a real difference to you?",
    "family_contact": "What about contact with your family or siblings?",
}
CHANGE_Q = "If you could change one thing about the care system, what would it be?"
PROBE_LIST = sorted(PROBES)

NUM = ["two", "three", "four", "five", "six", "seven", "eight"]
WAIT = ["a few months", "six months", "almost a year", "over a year", "nearly two years"]

CORE = {
    "placements": {
        "neg": ["I was moved {n} times and I never felt settled anywhere.",
                "Every time I started to feel at home I was packed up and sent somewhere new.",
                "I had {n} different foster families and my belongings kept getting left behind.",
                "Moves happened with almost no warning, sometimes on the same day.",
                "I was placed miles from my friends and my town, so I lost everyone I knew."],
        "pos": ["I stayed with the same foster family for years and it gave me a real sense of home.",
                "I only moved once and the move was explained to me properly beforehand.",
                "My placement felt stable, I knew where I would be sleeping from one month to the next.",
                "They kept me close to my school and friends when I had to move."]},
    "education": {
        "neg": ["I changed schools {n} times and fell behind in almost every subject.",
                "Nobody at school understood what was happening at home, so my grades dropped.",
                "I missed my exams because of a move and no one arranged for me to resit them.",
                "I wanted to go to college but nobody helped me with the applications.",
                "Teachers expected less from me because I was in care."],
        "pos": ["My teacher kept pushing me and I ended up getting into college.",
                "The virtual school team arranged a tutor for me and my grades improved.",
                "School was the one place that felt steady for me.",
                "I got extra help with my exams and it made a big difference."]},
    "social_worker": {
        "neg": ["My social worker changed every few months and I had to retell my story each time.",
                "I hardly ever saw my social worker because they cancelled most visits.",
                "They talked to the adults and never really to me.",
                "My social worker seemed overloaded with cases and was always in a rush.",
                "I never knew who to call when something went wrong."],
        "pos": ["My social worker visited regularly and remembered the little things about me.",
                "She was honest with me even when the news was hard.",
                "I had the same social worker for several years and I could trust them.",
                "They explained decisions to me in a way I could understand."]},
    "mental_health": {
        "neg": ["I waited {w} for counselling and by then I had stopped asking for help.",
                "I felt anxious and low all the time but the service had a long waiting list.",
                "Nobody asked how I was feeling, they only asked about my behaviour.",
                "I was told I was too young or too old for the service, so I was passed around.",
                "The support ended as soon as I turned eighteen."],
        "pos": ["I got counselling quickly and it helped me make sense of what had happened.",
                "My carer noticed I was struggling and arranged therapy straight away.",
                "There was a worker at school I could talk to whenever I needed.",
                "The service stayed with me after I left care, which I did not expect."]},
    "leaving_care": {
        "neg": ["At eighteen I was moved into a flat on my own and nobody explained how to pay bills.",
                "The support just stopped on my birthday and I had no idea where to turn.",
                "I was given a flat with no furniture and nobody to call when the heating broke.",
                "I could not cook or budget because no one had ever taught me.",
                "Leaving care felt like falling off a cliff with no gradual handover."],
        "pos": ["My personal adviser helped me find a flat and set up my bills before I left.",
                "I had a proper plan for leaving care and I knew what to expect.",
                "I was allowed to stay with my foster carer after eighteen, which made the transition easier.",
                "A leaving care team taught me to budget and cook before I moved out."]},
    "voice": {
        "neg": ["Decisions about where I lived were made in meetings I was not invited to.",
                "My views were written in the file but never seemed to change anything.",
                "Nobody asked me what I thought about my care plan, they just told me the outcome.",
                "When I spoke up in reviews, it felt like the adults had already made up their minds.",
                "I found out about big decisions after they had been made."],
        "pos": ["I was invited to my reviews and my opinion was taken seriously.",
                "My social worker asked what I wanted before every decision.",
                "I had an advocate who made sure my voice was heard.",
                "They changed my placement plan because of what I said."]},
    "trusted_adult": {
        "neg": ["Honestly there was nobody, I felt like I had to manage on my own.",
                "The adults around me kept changing, so I never built a bond with anyone.",
                "I had no one to turn to when things went wrong.",
                "People were paid to be there and I could tell it was just a job."],
        "pos": ["A youth worker stayed in my life for years and made the biggest difference.",
                "My foster carer was the one steady person and I still talk to her.",
                "A teacher believed in me when nobody else did.",
                "A mentor from a charity checked in on me every week."]},
    "family_contact": {
        "neg": ["I was only allowed to see my brother for an hour a month in an office.",
                "Contact with my family was cancelled without any explanation.",
                "I lost touch with my siblings because we were placed in different towns.",
                "Nobody helped me rebuild my relationship with my mum."],
        "pos": ["I saw my siblings every week and that kept us close.",
                "Social services supported contact with my gran and it meant a lot.",
                "We were placed together as siblings, which was a huge relief.",
                "They helped me stay in touch with my family safely."]},
}
ELAB = {
    "placements": {"neg": ["It made it hard to trust that anything would last.",
                           "I stopped unpacking my bag because I knew I would be moved again.",
                           "I still find it difficult to settle in one place."],
                   "pos": ["That stability made everything else easier.", "I still visit them now."]},
    "education": {"neg": ["I still feel behind compared to people my age.",
                          "Nobody talked to me about university or apprenticeships."],
                  "pos": ["I am now studying and proud of it.",
                          "Having someone who believed in me changed my attitude to learning."]},
    "social_worker": {"neg": ["It felt like I was just a case number.", "After a while I stopped opening up to anyone."],
                      "pos": ["That consistency really mattered to me.", "They treated me like a person, not a file."]},
    "mental_health": {"neg": ["I coped on my own and it took a toll on me.", "A crisis happened before anyone took it seriously."],
                      "pos": ["I felt listened to for the first time.", "It gave me tools I still use."]},
    "leaving_care": {"neg": ["I felt completely alone for the first year.", "I ran into debt because I did not understand the rules."],
                     "pos": ["I felt prepared and a bit less scared.", "They kept checking in during my first months."]},
    "voice": {"neg": ["It felt like my life was being managed, not lived.", "I stopped bothering to give my opinion."],
              "pos": ["Being heard gave me confidence.", "It showed me my opinion mattered."]},
    "trusted_adult": {"neg": ["I learnt not to rely on anyone.", "That loneliness is hard to explain."],
                      "pos": ["She stayed when everything else changed.", "Knowing someone cared gave me strength."]},
    "family_contact": {"neg": ["I still grieve those lost years.", "It felt like a punishment for something that was not my fault."],
                       "pos": ["Those relationships kept me going.", "I am still close to them."]},
}
RECS = {
    "placements": ["I would stop moving young people so often and keep siblings and friends close.",
                   "Moves should be planned and explained, not done overnight."],
    "education": ["I would give every young person a tutor and a named person at school.",
                  "Exams should never be missed because of a move."],
    "social_worker": ["I would make sure we keep the same social worker for years.",
                      "Social workers need smaller caseloads so they have time for us."],
    "mental_health": ["I would make counselling available straight away with no waiting list.",
                      "Support should carry on after eighteen."],
    "leaving_care": ["I would teach us about bills and budgeting before we leave and not cut support at eighteen.",
                     "There should be a gradual handover, not a cliff edge."],
    "voice": ["I would make sure every young person is in their own review meetings.",
              "Young people should have a real say and someone to speak up for them."],
    "trusted_adult": ["I would make sure everyone has at least one adult who stays in their life.",
                      "Mentors should be available to every young person."],
    "family_contact": ["I would protect contact with brothers and sisters.",
                       "Contact should be supported and not cancelled lightly."],
}
NOISE = ["I do not really remember much about that.", "It is hard to put into words.", "Yeah, I guess so.",
         "I would rather not go into it.", "I am not sure what to say about that."]

REGIONS = ["London", "North West", "West Midlands", "Scotland", "Wales", "South West"]
REGION_W = [0.25, 0.2, 0.15, 0.15, 0.1, 0.15]
AGES = ["18-21", "22-25", "26-30"]
TIME = ["under 2 years", "2-5 years", "over 5 years"]
ETH = ["White", "Black or Black British", "Asian or Asian British", "Mixed", "Other"]
ETH_W = [0.6, 0.14, 0.14, 0.08, 0.04]   # random on purpose: no planted effect


def fill(s):
    return s.format(n=rng.choice(NUM), w=rng.choice(WAIT))


def neg_prob(topic, stability, support, region):
    if topic in ("placements", "education", "leaving_care", "family_contact"):
        p = 0.8 if stability == "low" else 0.2
    elif topic in ("social_worker", "voice", "trusted_adult"):
        p = 0.8 if support == "low" else 0.2
    else:  # mental_health: planted regional effect
        p = 0.55 + (0.25 if region in ("North West", "Wales") else 0) + (0.1 if stability == "low" else 0)
    return min(p, 0.95)


def answer(topic, stance):
    if stance == "mixed":
        a, b = rng.choice(CORE[topic]["neg"]), rng.choice(CORE[topic]["pos"])
        return fill(a) + " That said, " + b[0].lower() + b[1:]
    text = fill(rng.choice(CORE[topic][stance]))
    if rng.random() < 0.3:
        text += " " + rng.choice(ELAB[topic][stance])
    return text


def main():
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    people, truth = [], []
    for i in range(1, N + 1):
        pid = f"P{i:03d}"
        region = rng.choices(REGIONS, REGION_W)[0]
        stability = "low" if rng.random() < 0.45 else "high"
        support = "low" if rng.random() < 0.5 else "high"
        people.append({"participant_id": pid, "region": region, "age_band": rng.choice(AGES),
                       "time_in_care": rng.choice(TIME), "ethnic_group": rng.choices(ETH, ETH_W)[0]})
        lines, weak = [f"# SYNTHETIC INTERVIEW {pid}"], []

        def add(speaker, text, topic=None, stance=None):
            lines.append(f"{speaker}: {text}")
            if speaker == "PARTICIPANT":
                truth.append({"participant_id": pid, "line_no": len(lines), "topic": topic, "stance": stance})

        for topic, q in QUESTIONS.items():
            if rng.random() < 0.1:
                continue   # question skipped (ran out of time)
            if rng.random() < 0.12:
                stance = "mixed"
            else:
                stance = "neg" if rng.random() < neg_prob(topic, stability, support, region) else "pos"
            if stance == "neg":
                weak.append(topic)
            add("INTERVIEWER", q)
            if rng.random() < 0.03:
                add("PARTICIPANT", rng.choice(NOISE), "none", "none")
                continue
            add("PARTICIPANT", answer(topic, stance), topic, stance)
            if rng.random() < 0.25:
                add("INTERVIEWER", rng.choice(PROBE_LIST))
                s = "neg" if stance == "mixed" else stance
                add("PARTICIPANT", rng.choice(ELAB[topic][s]), topic, stance)
        add("INTERVIEWER", CHANGE_Q)
        t = rng.choice(weak) if weak else rng.choice(list(RECS))
        add("PARTICIPANT", rng.choice(RECS[t]), t, "rec")
        (TRANSCRIPTS / f"{pid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path, rows in ((PARTICIPANTS, people), (GROUND_TRUTH, truth)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {N} transcripts and {len(truth)} participant answers")


if __name__ == "__main__":
    main()
