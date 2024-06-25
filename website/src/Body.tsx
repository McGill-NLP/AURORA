import fig1 from './assets/fig1.png'
import fig2 from './assets/fig2.png'

const Section = ({ title, hasTitle = true, children }: any) => {
  const marginClass = hasTitle ? 'my-10' : 'my-0';
  return (
    <section className={`${marginClass} flex flex-col`}>
      {hasTitle && <h2 className="border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0">{title}</h2>}
      {children}
    </section>
  );
};

export const Body = () => {
  return (
    <main className='px-5 py-5 lg:px-40 lg:py-10'>
      <Section title="Abstract" hasTitle={false}>
        <p className="leading-7 [&:not(:first-child)]:mt-6 text-justify">
            An image editing model should be able to perform diverse edits, ranging from object replacement and changing attributes or style to performing actions or movements, which require various forms of reasoning. Current general instruction-guided editing models have significant shortcomings with action and reasoning-centric edits. Object, attribute, or stylistic changes can be learned from visually static datasets. However, high-quality data for action and reasoning-centric edits is scarce and must come from entirely different sources that cover physical dynamics, temporality, and spatial reasoning.

            To address this, we meticulously curate the AROAD Dataset (Action-Reasoning-Object-Attribute), a collection of high-quality training data, human-annotated and curated from videos and simulation engines. We focus on a key aspect of quality training data: triplets (source image, prompt, target image) that contain a single meaningful visual change described by the prompt, ensuring truly minimal changes between source and target images.

            To demonstrate the value of our dataset, we evaluate an AROAD-finetuned model on a new expert-curated benchmark (AROABench) covering eight diverse editing tasks. Our model significantly outperforms previous editing models as judged by human raters. For automatic evaluations, we find important flaws in previous metrics and caution their use for semantically hard editing tasks. Instead, we propose a new automatic metric that focuses on discriminative understanding.

            We hope that our efforts in (1) curating a quality training dataset and an evaluation benchmark, (2) developing critical evaluations, and (3) releasing a state-of-the-art model will fuel further progress in general image editing.
          </p>
          <img src={fig1} alt="Figure 1" style={{height: "auto", width: "auto"}}/>
      </Section>
      <Section title="AURORA Dataset">
        <p className="leading-7 [&:not(:first-child)]:mt-6 text-justify">
          We present AURORA, a balanced dataset covering Action Reasoning, Object and Attribute edits, comprising a total of 289K training examples, see Fig. 2 and Tab. 1 for details and comparison to existing datasets. 
        </p>
        <img src={fig2} alt="Figure 2" style={{height: "auto", width: "auto"}}/>
      </Section>
      <Section title="AURORA-BENCH">
        <p className="leading-7 [&:not(:first-child)]:mt-6 text-justify">
          To holistically assess the editing abilities defined in Section 4 (object/attribute, global, action, reasoning, excluding viewpoint), we manually create a set of 400 image-edit-instruction pairs from 8 sources: AROABench. See Figure 5 for an example of each one. We ensure that AROABench allows studying out-of-distribution (OOD) transfer when a model is trained on AROAD, e.g., Sim2Real transfer from Kubric-Edit to real-world (spatial) reasoning or action edits outside of Action-Genome-Edit or Something-Something-Edit. Each of the 8 tasks contains 50 examples of image-prompt pairs that were either directly written by the authors or manually inspected for quality.
        </p>
      </Section>
      <Section title="Demo">
        <p className="leading-7 [&:not(:first-child)]:mt-6 text-justify">
          Demo: 
        </p>
        <iframe
          src="https://timbrooks-instruct-pix2pix.hf.space"
          width="auto"
          height="450"
        ></iframe>

      </Section>
    </main>
  )
}
