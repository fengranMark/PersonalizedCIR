import json

def preprocess(input_file, output_file):
    temp_q = ''
    temp_a = ''
    with open(input_file) as f:
        d = json.load(f)
        for conv in d:
            number = conv['number']
            title = conv['title']
            ptkb = conv['ptkb']
            for turn in conv['turns']:
                data = {}
                turn_id = turn['turn_id']
                data['sample_id'] = number + '-' + str(turn_id)
                data['number'] = number
                data['title'] = title
                data['cur_utt_text'] = turn['utterance']
                data['oracle_utt_text'] = turn['resolved_utterance']
                data['cur_response_text'] = turn['response']
                data['ptkb'] = ptkb
                data['ptkb_provenance'] = turn['ptkb_provenance']
                data['response_provenance'] = turn['response_provenance']
                if turn_id == 1:
                    ctx_utts_text = []
                    history_response = []
                else:
                    ctx_utts_text.append(temp_q)
                    history_response.append(temp_a)
                data['ctx_utts_text'] = ctx_utts_text
                data['history_response'] = history_response
                temp_q = turn['utterance']
                temp_a = turn['response']
                with open(output_file, 'a+') as file:
                    json.dump(data, file)
                    file.write('\n')


# preprocess('2023_test_topics.json', '2023_test_topics_new.jsonl')
