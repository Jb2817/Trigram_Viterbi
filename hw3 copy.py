
def parse_training_corpus(file_path):
  
    sentences = []  
    unique_words = set() 
    with open(file_path, 'r') as file:
        current_sentence = []
        for line in file:
            if line.strip():  # Non-empty line
                token, pos_tag = line.strip().split()
                current_sentence.append((token, pos_tag))
                unique_words.add(token)
            else:  # Empty line indicates end of a sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
    return sentences, unique_words


training_data, training_words = parse_training_corpus('WSJ_02-21.pos')


from collections import defaultdict, Counter
def precompute_probabilities(training_data, tag_list, word_tag_likelihoods, oov_strategy, training_words, word_distribution, total_tag_counts):
    word_tag_probs = {}
    for word in training_words:
        word_tag_probs[word] = {tag: word_tag_likelihoods.get((word, tag), oov_strategy(word, tag, word_distribution, total_tag_counts)) for tag in tag_list}
    return word_tag_probs

def calculate_probabilities(sentences):
    tag_frequency = Counter()
    word_tag_frequency = Counter()

    for sentence in sentences:
        for word, tag in sentence:
            tag_frequency[tag] += 1
            word_tag_frequency[(word, tag)] += 1

    total_tags = sum(tag_frequency.values())

    # Calculating prior probabilities of each POS tag
    tag_prior_probabilities = {tag: freq / total_tags for tag, freq in tag_frequency.items()}

    # Calculating likelihood of words given POS tags
    word_given_tag_likelihood = {word_tag: freq / tag_frequency[tag] 
                                 for word_tag, freq in word_tag_frequency.items()}

    return tag_prior_probabilities, word_given_tag_likelihood


def modified_viterbi(words, tag_list, tag_priors, word_tag_likelihoods, oov_strategy, transition_probabilities, word_distribution, total_tag_counts):
    if not words:
        return []

    # Initialize the dynamic programming table
    viterbi = [{}]
    path = {}

    # Initialize the first column of the Viterbi table
    for tag in tag_list:
        prob = tag_priors.get(tag, 0) * word_tag_likelihoods.get((words[0], tag), oov_strategy(words[0], tag, word_distribution, total_tag_counts))
        viterbi[0][tag] = prob
        path[tag] = [tag]

    # Run Viterbi for t > 0
    for t in range(1, len(words)):
        viterbi.append({})
        newpath = {}

        for tag in tag_list:
            (prob, state) = max(
                (viterbi[t-1][y0] * transition_probabilities.get((y0, tag), 0) * word_tag_likelihoods.get((words[t], tag), oov_strategy(words[t], tag, word_distribution, total_tag_counts)), y0)
                for y0 in tag_list
            )
            viterbi[t][tag] = prob
            newpath[tag] = path[state] + [tag]

        path = newpath

    n = len(words) - 1
    (prob, state) = max((viterbi[n][tag], tag) for tag in tag_list)
    return path[state]



def oov_strategy(word, tag, word_distribution, total_tag_counts):
    # Hard-coded probabilities based on features
    if word.endswith('s'):
        return 0.5 if tag == 'NNS' else 1/1000
    elif word[0].isupper():
        return 0.5 if tag == 'NNP' else 1/1000
    elif not word.isalnum():
        return 0.5 if tag in {'#', '$', '.', ',', ':', '('} else 1/1000
    elif word.isdigit():
        return 1.0 if tag == 'CD' else 1/1000

    # Distribution of items occurring once
    if word_distribution.get(tag, 0) > 0:
        return word_distribution[tag] / total_tag_counts[tag]

    # Default OOV strategy
    return 1/1000

def calculate_word_distribution(training_data):
    word_distribution = Counter()
    total_tag_counts = Counter()

    for sentence in training_data:
        for word, tag in sentence:
            word_distribution[word] += 1
            total_tag_counts[tag] += 1

    return word_distribution, total_tag_counts
    single_occurrence_words = {word for word, freq in word_freq.items() if freq == 1}

   
    single_word_tag_distribution = Counter({tag: 0 for tag in tag_counts})
    for sentence in training_data:
        for word, tag in sentence:
            if word in single_occurrence_words:
                single_word_tag_distribution[tag] += 1

    return single_word_tag_distribution, tag_counts



def calculate_transition_probabilities(sentences):
    
    # Counters for bigram frequencies and tag frequencies
    bigram_freq = Counter()
    tag_freq = Counter()

    for sentence in sentences:
        # Add a start symbol '<s>' to the beginning of each sentence
        tags = ['<s>'] + [tag for _, tag in sentence]
        for i in range(len(tags) - 1):
            tag_freq[tags[i]] += 1
            bigram_freq[(tags[i], tags[i+1])] += 1

    # Calculate transition probabilities
    transition_prob = {bigram: count / tag_freq[bigram[0]] for bigram, count in bigram_freq.items()}

    return transition_prob
from tqdm import tqdm

START_SYMBOL = '<s>'
STOP_SYMBOL = '</s>'
def main():
    # Parse the training data
    training_data, training_words = parse_training_corpus('WSJ_02-21.pos')

    # Calculate tag priors and word-tag likelihoods
    tag_priors, word_tag_likelihoods = calculate_probabilities(training_data)

    # Calculate word distribution and total tag counts for OOV strategy
    word_distribution, total_tag_counts = calculate_word_distribution(training_data)

    # Prepare tag list and include special symbols
    unique_tags = set(tag for sentence in training_data for _, tag in sentence)
    tag_list = ['<s>', '</s>'] + list(unique_tags)

    # Precompute word-tag probabilities
    word_tag_probs = precompute_probabilities(training_data, tag_list, word_tag_likelihoods, oov_strategy, training_words, word_distribution, total_tag_counts)

    # Process the development test file (WSJ_24.words)
    with open('WSJ_24.words', 'r') as file:
        dev_test_sentences = [line.strip().split() for line in file.read().split('\n\n')]

    # Predict the POS tags for each sentence
    predicted_output = []
    for words in tqdm(dev_test_sentences):
        if not words:  # Skip empty sentences
            continue
        # Directly pass the additional arguments to oov_strategy
        predicted_tags = modified_viterbi(words, tag_list, tag_priors, word_tag_probs, 
                                          oov_strategy, calculate_transition_probabilities(training_data),
                                          word_distribution, total_tag_counts)
        predicted_output.extend([f"{word}\t{tag}" for word, tag in zip(words, predicted_tags)] + [''])
    # Ensure an empty line at the end of the file
    if predicted_output and predicted_output[-1] != '':
        predicted_output.append('')

    # Write the predictions to a file
    with open('predicted_tags.pos', 'w') as file:
        file.write('\n'.join(predicted_output))

if __name__ == "__main__":
    main()
