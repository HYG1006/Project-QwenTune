import torch
import torch.nn.functional as F
import math

import model
from data import Corpus

device = 'cuda'
corpus = Corpus(path="../data/ptb",
                batch_size={'train':50, 'valid':10},
                max_sql=256)

# load all models
lstm = model.LMModel_LSTM(nvoc=10000, dim=256, num_layers=2, hidden_size=256, dropout=0.4538)
rnn = model.LMModel_RNN(nvoc=10000, dim=256, num_layers=2, hidden_size=256, dropout=0.5756)
transformer = model.LMModel_transformer(nvoc=10000, dim=256, num_layers=4, hidden_size=512, dropout=0.2008)
ckpt = torch.load('ckpts/lstm_50.pth', map_location='cpu')
lstm.load_state_dict(ckpt)
lstm.eval().to(device)
ckpt = torch.load('ckpts/transformer_50.pth', map_location='cpu')
transformer.load_state_dict(ckpt)
transformer.eval().to(device)
ckpt = torch.load('ckpts/rnn_50.pth', map_location='cpu')
rnn.load_state_dict(ckpt)
rnn.eval().to(device)
print('load model finished')

def generate(model, prompt, max_len=50, temperature=1.0):
    # sample func. for LSTM && RNN
    model.eval()
    words = prompt.split()
    idxs = [corpus.word_id.get(w, corpus.word_id['<unk>']) for w in words]
    input_ids = torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(1)
    hidden = None
    generated = words.copy()
    generated_ids = idxs.copy()
    with torch.no_grad():
        if input_ids.size(0) > 1:
            out, hidden = model(input_ids[:-1], hidden)
        prev_id = input_ids[-1].unsqueeze(0)
        for _ in range(max_len):
            # forward
            logits, hidden = model(prev_id, hidden)
            logits = logits.squeeze(0).squeeze(0)
            # apply temp.
            probs = F.softmax(logits / temperature, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)
            next_word = corpus.vocabulary[next_id.item()]
            generated.append(next_word)
            generated_ids.append(next_id.item())
            prev_id = next_id.view(1,1)
            if next_word == '<eos>':
                break
    return ' '.join(generated)

def generate_transformer(model, prompt, max_len=50, temperature=1.0):
    # sample func. for transformer
    model.eval()
    words = prompt.split()
    idxs = [corpus.word_id.get(w, corpus.word_id['<unk>']) for w in words]
    generated = idxs.copy()
    with torch.no_grad():
        for _ in range(max_len):
            input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            outputs, _n = model(input_ids)
            logits = outputs[0, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
            if corpus.vocabulary[next_id] == '<eos>':
                break
    return ' '.join(corpus.vocabulary[i] for i in generated)


def calc_sentence_ppl(model, sentence):
    model.eval()
    with torch.no_grad():
        tokens = sentence.strip().split()
        ids = [corpus.word_id.get(tok, corpus.word_id['<unk>']) for tok in tokens]
        ids_in  = ids
        ids_out = ids[1:] + [corpus.word_id['<eos>']]
        input_ids = torch.LongTensor(ids_in).unsqueeze(1).to(device)
        target    = torch.LongTensor(ids_out).to(device)
        decoded, _ = model(input_ids)
        logits = decoded.view(-1, decoded.size(-1))
        loss = F.cross_entropy(logits, target, reduction='sum')
        ppl = math.exp(loss.item() / target.size(0))
    return ppl


sentences_of_wiki_article = ['This article is about the present-day human-induced rise in global temperatures',
                             'Global warming redirects here changes in surface air temperature over the past 50 years',
                             'Earth\'s average surface air temperature has increased almost 1.5 °C (about 2.5 °F) since the Industrial Revolution',
                             'Present-day climate change includes both global warming—the ongoing increase in global average temperature—and its wider effects on Earth\'s climate system',
                             'The current rise in global temperatures is driven by human activities especially fossil fuel burning since the Industrial Revolution Fossil fuel use deforestation and some agricultural and industrial practices release greenhouse gases'
                             'These gases absorb some of the heat that the Earth radiates after it warms from sunlight warming the lower atmosphere Carbon dioxide the primary gas driving global warming has increased in concentration by about 50 since the pre-industrial era to levels not seen for millions of years',
                             'Deserts are expanding while heat waves and wildfires are becoming more common',
                             'Amplified warming in the Arctic has contributed to thawing permafrost retreat of glaciers and sea ice decline',
                             'Higher temperatures are also causing more intense storms droughts and other weather extremes',
                             'Rapid environmental change in mountains coral reefs and the Arctic is forcing many species to relocate or become extinct even if efforts to minimize future warming are successful some effects will continue for centuries',
                             'These include ocean heating ocean acidification and sea level rise',
                             'Climate change threatens people with increased flooding extreme heat increased food and water scarcity more disease, and economic loss',
                             'Human migration and conflict can also be a result the World Health Organization calls climate change one of the biggest threats to global health in the 21st century',
                             'Societies and ecosystems will experience more severe risks without action to limit warming adapting to climate change through efforts like flood control measures or drought-resistant crops partially reduces climate change risks although some limits to adaptation have already been reached',
                             'Effects of climate change are well documented and growing for Earth\'s natural environment and human societies',
                             'Changes to the climate system include an overall warming trend changes to precipitation patterns and more extreme weather',
                             'As the climate changes it impacts the natural environment with effects such as more intense forest fires thawing permafrost and desertification',
                             'These changes impact ecosystems and societies and can become irreversible once tipping points are crossed',
                             'Climate activists are engaged in a range of activities around the world that seek to ameliorate these issues or prevent them from happening',
                             'Climate change mitigation (or decarbonisation) is action to limit the greenhouse gases in the atmosphere that cause climate change',
                             'Climate change mitigation actions include conserving energy and replacing fossil fuels with clean energy sources secondary mitigation strategies include changes to land use and removing carbon dioxide (CO₂) from the atmosphere',
                             'Current climate change mitigation policies are insufficient as they would still result in global warming of about 2.7 °C by 2100, significantly above the 2015 Paris Agreement\'s goal of limiting global warming to below 2 °C',
                             'Solar energy and wind power can replace fossil fuels at the lowest cost compared to other renewable energy options',
                             'The availability of sunshine and wind is variable and can require electrical grid upgrades such as using long-distance electricity transmission to group a range of power sources',
                             'Energy storage can also be used to even out power output and demand management can limit power use when power generation is low',
                             'Cleanly generated electricity can usually replace fossil fuels for powering transportation heating buildings and running industrial processes',
                             'Certain processes are more difficult to decarbonise such as air travel and cement production',
                             'Carbon capture and storage (CCS) can be an option to reduce net emissions in these circumstances although fossil fuel power plants with CCS technology is currently a high cost climate change mitigation strategy',
                             'Climate change adaptation is the process of adjusting to the effects of climate change both current and anticipated',
                             'Adaptation aims to moderate or avoid harm for people and is usually done alongside climate change mitigation',
                             'It also aims to exploit opportunities',
                             'Humans may also intervene to help adjust for natural systems',
                             'There are many adaptation strategies or options',
                             'Some examples of these are building seawalls or inland flood defenses providing new insurance schemes changing crop planting times or varieties and installing green roofs or green spaces',
                             'Adaptation can be reactive or proactive',
                             'The need for adaptation varies from place to place',
                             'Adaptation measures vary by region and community depending on specific climate impacts and vulnerabilities scientific research into climate change adaptation generally starts with analyses of the likely effects of climate change on people ecosystems and the environment',
                             'Impacts may include changed agricultural yields increased floods and droughts or coral reef bleaching',
                             'As of 2022 the level of warming is 1.2 °C above levels before the industrial revolution'
                             'It is on track to increase to 2.5 to 2.9 °C by the end of the century this is causing a variety of secondary effects'
                            ]           


# evaluate ppl on wiki-article
for sent in sentences_of_wiki_article:
    ppl = calc_sentence_ppl(transformer, sent)
    print(f"句子：{sent}\nPerplexity = {ppl:.2f}\n")

# exit()
prompt = "the meaning of life is"
text = generate(rnn, prompt, max_len=50, temperature=0.8)
print('RNN: ', text)
text = generate(lstm, prompt, max_len=50, temperature=2)
print('LSTM: ', text)
text = generate_transformer(transformer, prompt, max_len=50, temperature=1.6)
print('Transformer: ', text)

# low temp.
# rnn: the meaning of life is now <eos>
# lstm: the meaning of life is the biggest market <eos>
# trans: the meaning of life is putting what is able to a specialty and about fiscal N million from other defense <eos>

# high temp.
# rnn: the meaning of life is new days forced easier change because that attacks <eos>
# lstm: the meaning of life is obviously seeing government <unk> now accountants sought production it reached managers ' <unk> tools using rumors ranging money but healthy billions china looks featuring glass work <unk> affecting television deductions activities slashing property movements papers last icahn in last aggressive <eos>
# trans: the meaning of life is buying george and economics instead as movie plus one imported bags a five-year compensation of bribery totaling # million works from campeau sells discussions on electronics services disposable items next fiscal in highly court-appointed colorado stepped worrying passed male in rate of securities agency yen because patients tissue segment operating