📑 Implementation-Ready Variable Glossary

Exactly what each variable must contain, how it is formatted, where it is generated, and where it is later consumed.

Variable	Data-type & Required Format	Authoritative Producer	Down-stream Consumers	Notes / Construction Rules
agent_name	str – full ecclesiastical name, e.g. "Cardinal Giovanni Rossi" (no titles like “His Eminence”; include “Cardinal”)	static config / roster file	every prompt	Must be unique primary key for agent-lookup.
biography	multiline str – raw text block with paragraph breaks	dataset loader	INTERNAL_PERSONA_EXTRACTOR	Nothing is stripped; preserve original punctuation.
persona_internal	str (markdown) – exactly 4 bullet lines produced by Internal Extractor:• ...	INTERNAL_PERSONA_EXTRACTOR	EXTERNAL_PROFILE_GENERATOR · DISCUSSION_REFLECTION · STANCE · VOTING_*	Must remain 4 bullets; no extra newlines after last bullet.
profile_public	str – max 2 sentences, neutral tone	EXTERNAL_PROFILE_GENERATOR	group assembly → group_profiles	No line-breaks; end with a period.
group_profiles	multiline str – exactly 5 lines, one per participant:Name – <profile_public> (<relation>)	grouping engine	DISCUSSION_*	Relation label from participant_relation_list.
participant_relation_list	str – semicolon-separated list, order matches group_profiles, e.g."Giovanni-sympathetic; Maria-neutral; ..." 	distance mapper	DISCUSSION_*	Labels limited to sympathetic, neutral, opposed.
discussion_round	int ≥ 1	scheduler	DISCUSSION_*	Increment each new small-group chat.
group_transcript	str – complete raw chat text of current group round	chat logger	DISCUSSION_ANALYZER	Preserve speaker tags if present.
group_summary	JSON object exactly:{"key_points":[str,str,str], "speakers":[str,str,str], "overall_tone":"..."}	DISCUSSION_ANALYZER	DISCUSSION_REFLECTION	Keys & lists must exist even if padded with "N/A".
agent_last_utterance	str – verbatim text of this agent’s most recent speech	chat logger	DISCUSSION_REFLECTION	Exclude system prompts.
reflection_digest	str – ≤ 25 words, single sentence, no bullet	DISCUSSION_REFLECTION	STANCE	Must not include line-breaks or quotes.
compact_scoreboard	str – semicolon list, each item:"<Name>:<votes> (<tag>)"Example: "A:12 (gaining); B:9 (stalling)"	tally function	DISCUSSION_* · DISCUSSION_REFLECTION · STANCE · VOTING_*	Tag algorithm: Δ≥+2 → gaining, Δ≤−2 → losing, otherwise stalling. Keep order by vote count desc.
visible_candidates	list[str] – JSON-style array or Python list literal	momentum filter	DISCUSSION_* · STANCE · VOTING_*	Round 1 = full roster; subsequent rounds = top-N (e.g. 5) by votes.
threshold	int – absolute number of votes required for ⅔ majority	calc at simulation start	DISCUSSION_* · DISCUSSION_REFLECTION · STANCE · VOTING_*	Re-compute if electorate size changes mid-run.
voting_round	int ≥ 1	scheduler	STANCE · VOTING_*	Increment every ballot.
role_tag	str – "CANDIDATE" or "ELECTOR"	static config	STANCE	Case-sensitive; no other values.
stance_digest	str – single sentence (<= 25 w) summarising current preference	STANCE	DISCUSSION_* · VOTING_*	Must contain exactly one candidate name from visible_candidates.

Parsing tip: treat every variable as immutable once produced in its stage; never mutate in-place except compact_scoreboard which is regenerated each tally.