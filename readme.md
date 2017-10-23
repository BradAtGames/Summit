# Summit

Summit is an implementation of TextRank based summarization done as a minor project for my research proficiency exam. TextRank is used to compute an overall importance score on a per sentence basis. We then apply this calculated score for an extractive summarization technique in which sentences deemed important are extracted from the original text and output as a summarized text in their original order.

The text is first transformed into a weighted, undirected graph. The weight of any given edge is equal to the similarity measure of the pair of sentences and the importance of a node is considered to be proportional to the sum of the weights of all of its incident edges. Edges with weight below a predetermined threshold are dropped altogether.

A more in-depth explanation can be found in [TextRank: Bringing Order to Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

I have included a copy of my python environment in [env.txt](env.txt). The majority of the environment is probably not needed to run the application. Installing nltk, scipy, and numpy should get all of the necessary dependencies.

### Example Run

>python main.py --text=gettysburg.txt

#### Original text
>Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.
>Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.
>But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth.

#### Summarized Text
>We are met on a great battle-field of that war.
>The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract.
>It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
