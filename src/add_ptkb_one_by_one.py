import json
import openai
import time
import re
import random



def main():
    args = get_args()

    # OpenAI API key
    openai.api_key = ""
    model_engine = "gpt-3.5-turbo-16k"
    set_176 = {'14-1-8', '10-1-9', '10-1-20', '17-3-4', '12-1-2', '17-1-10', '10-2-10', '15-2-4', '16-1-3', '16-1-14', '10-2-3', '18-2-8', '20-2-2', '14-2-5', '16-1-6', '20-2-9', '15-1-15', '15-2-2', '14-1-7', '11-2-3', '19-1-5', '10-3-7', '9-2-5', '12-1-5', '20-1-4', '17-3-3', '13-1-5', '9-1-5', '17-1-5', '11-2-7', '20-1-8', '20-1-9', '10-1-7', '14-2-6', '9-2-11', '17-2-2', '20-1-12', '16-1-9', '17-3-6', '10-3-9', '16-1-13', '15-2-10', '20-2-5', '12-1-4', '10-1-11', '18-1-12', '18-1-8', '9-2-2', '9-2-3', '18-1-5', '12-2-11', '15-1-12', '12-2-2', '14-1-17', '16-1-12', '19-1-15', '15-1-13', '17-2-12', '14-2-4', '14-2-11', '15-2-6', '19-1-7', '14-1-11', '11-1-7', '20-2-14', '13-1-3', '17-2-4', '9-2-9', '14-1-9', '15-1-6', '14-2-8', '20-1-11', '18-1-14', '17-1-7', '18-1-9', '10-1-17', '20-1-7', '11-2-2', '18-2-3', '12-1-6', '17-1-11', '14-1-12', '11-1-6', '17-2-5', '17-2-13', '18-2-6', '19-1-12', '9-1-6', '9-2-4', '16-1-7', '14-1-18', '12-1-9', '14-1-3', '15-1-14', '11-2-8', '15-2-1', '11-1-3', '14-1-4', '12-1-10', '14-1-14', '14-2-9', '11-1-9', '10-2-6', '13-1-6', '18-1-11', '19-1-3', '11-1-4', '9-1-3', '15-1-3', '20-2-4', '20-2-7', '17-2-11', '17-2-7', '10-1-2', '14-1-19', '13-1-1', '11-1-5', '15-2-11', '14-2-3', '18-1-10', '21-1-3', '20-2-6', '16-1-10', '19-1-11', '17-1-9', '10-1-15', '14-2-10', '16-1-8', '14-1-15', '16-1-4', '9-1-4', '12-2-8', '10-3-5', '11-1-8', '19-1-2', '21-1-1', '10-2-5', '12-2-3', '16-1-5', '14-2-7', '12-2-6', '10-2-7', '14-1-5', '10-3-6', '10-1-12', '19-1-14', '10-1-3', '12-2-10', '12-2-5', '11-2-11', '14-1-13', '18-1-6', '18-1-15', '15-1-2', '17-2-15', '20-1-6', '21-1-2', '12-2-9', '17-1-4', '14-2-12', '17-2-9', '20-2-3', '19-1-17', '14-2-2', '19-1-6', '18-2-12', '20-2-11', '10-1-6', '18-2-7', '9-2-7', '10-1-14', '15-2-9', '18-1-3', '10-2-13', '20-1-5', '15-1-5'}

    with open(args.input_path) as f:
        for line_num,line in enumerate(f):
            if line_num < 0 :
                continue
            data = json.loads(line)
            sample_id = data.get('sample_id','')
            if sample_id not in set_176:
                continue
            ptkb = data.get('ptkb', '')
            cur_utt = data.get('cur_utt_text','')
            cur_resp = data.get('cur_response_text','')
            conv = data.get('ctx_utts_text','')
            response_list = data.get('response', '')
            provenance = data.get('ptkb_provenance','')
            r_utterance = data.get('oracle_utt_text','')
            history_response = data.get('history_response','')
            turn = data.get('number', '')
            last_response = data.get('last_response','')


            for i,v in ptkb.items():
                if conv == []:
                    conv = cur_utt
                YourTask = '#Your Task:\n\n' + 'User\'s personal information: ' + v + '\n\n'
                for q,a in zip(conv,history_response):
                    YourTask += f'Question: {q}\nResponse: {a}\n\n'
                YourTask += 'Current Question: ' + cur_utt + '\n\n'

            
                prompt = f"""For an information-seeking dialog, please help reformulate the question into rewrite that can fully express the user's information needs without the need of context, but also generate an informative response to answer the question. You can generate a rewrite and response based on user's personal information.   
    """ + YourTask +'(Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**.Please provide a complete informative response, but keep it under 200 words. The output format should always be: Rewrite: $Rewrite\nResponse: $Response. Go ahead!)'
                print(sample_id)
                print(prompt)
                r = openai.ChatCompletion.create(
                        model = model_engine,
                        messages = [{"role":"system","content":"you are a query rewriter and knowledge selector"},{"role":"user","content":prompt}])

                cnv = r.choices[0].message["content"]
            
                print('——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————')
                print(cnv)

                match_rewrite = re.search(r'Rewrite:(.*?)Response:', cnv,re.DOTALL)
                match_response = re.search(r'Response:\s*(.*)', cnv,re.DOTALL)
                rewrite = match_rewrite.group(1)
                response = match_response.group(1)

                data['rewrite_utt_text'] = rewrite
                data['response_utt_text'] = response
                data['ptkb_num'] = i
                data['sample_id'] = sample_id + '-' +str(i)
                print(data['sample_id'])


                with open(args.output_path, 'a+') as outfile:
                        json.dump(data, outfile)
                        outfile.write('\n')
                time.sleep(3)  #  1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_new_filted.jsonl")
    parser.add_argument('--output_path', type=str, default="2023_test_output.jsonl")
    args = parser.parse_args()
    return args


        

if __name__ == '__main__':
    main()

