import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from longformer.longformer_encoder_decoder import (
    LongformerSelfAttentionForT5,
    LongformerT5Config,
    LongformerT5ForConditionalGeneration,
)


tokenizer = T5Tokenizer.from_pretrained("t5-base")
# model = LongformerT5ForConditionalGeneration.from_pretrained(
#     "/net/nfs2.s2-research/haokunl/exp_files/model_artifacts/t5/longt5-base-16384"
# )
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.eval()
model.config.gradient_checkpointing = True
model.encoder.config.gradient_checkpointing = True
model.decoder.config.gradient_checkpointing = True

TXT = """
Mayor Bill de Blasio will change a rule that has, for months, created a paradox in New York City’s school reopening plan: Classrooms that had been reopened to students often closed again because school buildings had to shut temporarily whenever two unrelated virus cases were detected.
The mayor announced Monday that he would alter the rule, but he did not explain how. He said the <extra_id_0> will be outlined in the coming days, but did not commit to making changes this week.
The closure rule has been extremely frustrating for many parents, who have said that every day brings uncertainty about whether their children will be able to attend school the following morning. Many schools have closed multiple times and sometimes have been open for just a few days before the next closure. The rule has also been intensely disruptive for educators, who have been forced to toggle between in-person and online learning with only a few hours’ notice.
The controversy over the closure rule has highlighted the enormous difficulties and trade-offs inherent in reopening schools during the pandemic. Mayors and education leaders across the country have scrambled to find ways to return students to classrooms while experimenting with safety protocols in real time.
Closures have accelerated in recent weeks and months, as middle and high school students have returned to their buildings after months of all-remote learning. The vast majority of New York City students — roughly 700,000 out of 1 million — have chosen to learn remotely full time, which means the closure rule did not affect most families.
But the city is giving all families an opportunity to switch from remote learning to classroom instruction for the rest of the school year, so that number may shift. Some students will get full-time instruction, while others will go in a few days a week and learn from home the rest of the time, based on individual school capacity. Families have until the end of the day on Friday to switch.
In recent weeks, some epidemiologists and medical experts have told ProPublica and the education news site Chalkbeat that New York’s two-case rule was arbitrary and had led to unnecessary closures, and called on the mayor to adjust it.
“The way to beat Covid is not by closing schools excessively, but by suppressing transmission both inside and outside of schools,” Dr. Dave A. Chokshi, the city’s health commissioner, said during a news conference on Monday.
The city’s schools have had very low virus transmission in classrooms since they began to reopen last fall. Michael Mulgrew, president of the United Federation of Teachers, has strenuously opposed any changes to the rule for months, arguing that the city’s schools were safe only because of the strict <extra_id_1>, including the two-case threshold.
“We can’t just say because they’re an inconvenience we don’t want them,” Mr. Mulgrew said of the guidelines during a radio interview last month.
The closure rule was settled last summer during a period of intense turmoil between City Hall and the union, at a moment when it was unclear whether Mr. de Blasio would be able to reopen schools at all. The city and union eventually agreed on a host of safety rules that cleared a path for New York to become the first large school district in America to reopen schools for all grades.
Several of those rules have changed over the last eight months. The mayor said over the summer, when the average citywide test positivity rate was hovering under 1 percent, that the entire school system would shut if the positivity rate hit 3 percent, which it did in November. He closed the school system for several weeks but came under significant pressure from parents and experts to set a different threshold.
When Mr. de Blasio reopened schools for young children and some students with disabilities in December, he said there would no longer be a citywide positivity threshold to shut the school system.
The city is also poised to partially change a rule it set over the summer that mandated six feet of distance between students in classrooms. Last month, the Centers for Disease Control and Prevention said districts should consider <extra_id_2> to three feet, a standard that Mr. de Blasio said the city would adopt in elementary school classrooms later this month.
That shift rankled the teachers’ union, which has had significant influence over the school reopening process in <extra_id_3>. Though relations between City Hall and the union have been frosty for months, the mayor has tried to maintain some peace with Mr. Mulgrew.
For example, when the city reopened elementary schools late last year despite rising virus cases across the city, Mr. de Blasio announced increased random testing in school buildings, a consistent union priority that experts have supported.
But the city and union have struggled to find a compromise on the two-case rule. For weeks, Mr. de Blasio said a revision to the rule was imminent, but behind the scenes, negotiations between the two sides were stalling. The city and union still do not have an agreement on what the new closure threshold should be.
While the mayor has the power to unilaterally change the rule, City Hall has tried to avoid alienating the union with just a few months left in the school year. The U.F.T. raised many issues with the city over the reopening plan last summer, but it has been more willing to reopen schools than some other teachers’ unions in big cities, including Chicago and Los Angeles.
The fact that all grades of the school system are open means that the union has less leverage now than at any point in the school reopening process. But the union still has enormous influence over how the next school year will unfold.
Mr. de Blasio has said he expects full-time, in-person instruction come September, though it is likely that there will be a remote option for some families into the fall. That goal will rest in part on the union’s cooperation and support, and teachers will no doubt play a crucial role in reaching out to reluctant families and encouraging them to return to classrooms.
"""
LABELS_1 = "<extra_id_0> new rules <extra_id_1> safety measures <extra_id_2> reducing the distance <extra_id_3> New York <extra_id_4>"
LABELS_0 = "<extra_id_0> full-time <extra_id_1> in-person instruction <extra_id_2> closing schools <extra_id_3> City Hall <extra_id_4>"
data = tokenizer([TXT], return_tensors="pt", padding="max_length", max_length=4096)
input_ids = data["input_ids"]
attention_mask = data["attention_mask"]
labels_1 = tokenizer(LABELS_1, return_tensors="pt").input_ids
labels_0 = tokenizer(LABELS_0, return_tensors="pt").input_ids
with torch.no_grad():
    loss_1 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, labels=labels_1)[0]
    loss_0 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, labels=labels_0)[0]

print("real", loss_1, "false", loss_0)
