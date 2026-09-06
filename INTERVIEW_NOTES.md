# Interview Notes — AI Data Cleaning & Preprocessing Agent

Use this file to prepare talking points. Everything here is written so
you can explain it confidently as a fresher, without needing to have
memorized advanced ML theory.

---

## 1. Why "Hybrid" (Deterministic + LLM)?

**Deterministic layer (Pandas/Scikit-learn):**
- Same input always produces the same output.
- Fast, cheap, and doesn't depend on an external API.
- Handles the *mechanical* parts of cleaning: renaming columns,
  removing duplicates, filling missing values, scaling numbers.

**LLM layer (OpenAI):**
- Good at *language* tasks: summarizing what happened in plain
  English, spotting patterns a human would describe qualitatively
  ("this column looks like it holds job titles"), and giving
  suggestions.
- Non-deterministic and slower/costlier, so it should never be the
  thing that actually transforms your numbers — only the thing that
  *explains* or *recommends*.

**One-line answer if asked "why not do everything with the LLM?"**
> Because core data transformations need to be reliable and
> reproducible — you don't want a model to "creatively" round or
> reinterpret your numbers. The LLM is there to communicate, not to
> compute.

---

## 2. Median vs. Mean Imputation

- **Mean** is sensitive to outliers. If most values are ~30 but one
  row is 10,000, the mean gets dragged way up and every imputed value
  becomes unrealistic.
- **Median** is the middle value when sorted, so a few extreme values
  don't distort it. This is why the project uses median for numeric
  columns — real-world data (salaries, transaction amounts, ages with
  data-entry errors) is often skewed.
- **Mode** is used for categorical columns because "average category"
  doesn't make sense — you fill with whichever category appears most
  often.

**Trade-off:** median/mode imputation is simple and safe but it does
reduce variance in the data (every missing value collapses to the same
number), which can slightly understate uncertainty. More advanced
techniques (KNN imputation, regression imputation) preserve more
signal but are heavier to justify for a first project.

---

## 3. IQR Winsorization vs. Deleting Rows

**IQR (Interquartile Range) method:**
1. Q1 = 25th percentile, Q3 = 75th percentile.
2. IQR = Q3 − Q1.
3. Anything below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR` is
   considered an outlier.

**Winsorization (capping)** clips those outliers to the fence value
instead of deleting the row. So a value of 10,000 in an "age" column
gets capped to, say, 85 (the upper fence) rather than the whole row
being thrown away.

**Why capping instead of deleting?**
- Deleting rows loses potentially valid information in *other*
  columns of that row.
- Capping keeps the dataset size constant, which matters when you
  don't have much data to begin with.
- It's a defensible middle ground between "ignore outliers" and
  "throw away rows".

---

## 4. Multicollinearity & the Dummy Variable Trap

When you one-hot encode a categorical column with *k* categories, you
normally get *k* new binary columns. But if you know the values of
*k-1* of them, you can always figure out the last one (if none of the
others are 1, the last one must be 1). That perfect redundancy is
**multicollinearity** — it can break certain models (like linear
regression) because the math behind them assumes features are
independent.

**Fix:** `drop_first=True` drops one dummy column per categorical
feature, leaving *k-1* columns. No information is lost — the dropped
category becomes the "baseline" that all other dummies are compared
against.

---

## 5. Graceful Degradation

The `ai_agent.py` module checks whether a valid `OPENAI_API_KEY` is
present *before* trying to call the API. If it's missing, empty, or
still the placeholder value, the functions immediately return a
rule-based summary built from the metadata dictionary instead. This
means:
- The app never crashes due to a missing/invalid key.
- A recruiter or interviewer can run the whole project with zero
  API cost.
- It demonstrates defensive, production-minded coding.

---

## 10 Interview Questions & Model Answers

**Q1. Walk me through what happens when a user uploads a CSV in this app.**
> The file is read into a Pandas DataFrame. It goes through cleaning
> (standardizing column names, removing duplicates, normalizing nulls,
> imputing missing values), then preprocessing (capping outliers with
> IQR, one-hot encoding categories, scaling numeric features), and
> finally an AI layer summarizes the whole run. Each stage is shown as
> a separate tab in the Streamlit UI with before/after metrics.

**Q2. Why do you standardize column names to snake_case?**
> Different data sources use different naming conventions
> ("First Name", "firstName", "first_name"). Standardizing avoids bugs
> from inconsistent column references later in the pipeline and makes
> the data easier to query in SQL, where snake_case is the convention.

**Q3. Why do you exclude ID columns from deduplication comparisons?**
> Because IDs are meant to be unique per row by design. If you include
> them when comparing rows for duplicates, two rows that are otherwise
> identical would never be flagged as duplicates just because their ID
> differs — defeating the purpose of deduplication.

**Q4. What's the difference between StandardScaler and Min-Max scaling?**
> StandardScaler (z-score) transforms values to have mean 0 and
> standard deviation 1, and is less sensitive to outliers than Min-Max
> scaling, which squashes values into a fixed [0,1] range using the
> min and max — so a single outlier can compress everything else into
> a tiny range. I use StandardScaler here because it pairs well with
> IQR capping, which already tames extreme values.

**Q5. Why exclude binary/target columns from scaling?**
> Scaling a column that's already 0/1 (like a "survived" label or a
> one-hot encoded dummy) doesn't help and can make the labels harder
> to interpret. Target variables usually need to stay in their
> original form for the model or metric to make sense.

**Q6. How does the FastAPI backend differ from the Streamlit app in this project?**
> They call the exact same underlying functions in `scripts/`. The
> Streamlit app is for interactive human demos; FastAPI exposes the
> same logic as JSON REST endpoints so another service (or a frontend
> built by someone else) could call the pipeline programmatically.
> This separation of "business logic" from "presentation" is a common
> real-world pattern.

**Q7. What happens if the OpenAI API key is invalid or the API call fails?**
> The `ai_agent.py` module wraps API calls in a try/except. If the key
> is missing or the call throws an exception, it falls back to a
> rule-based summary built directly from the metadata — so the rest of
> the app keeps working.

**Q8. Why use IQR-based capping instead of Z-score based outlier detection?**
> Z-score assumes roughly normal distribution and is itself sensitive
> to outliers (since it uses the mean and standard deviation). IQR is
> based on percentiles, which are robust to extreme values — a more
> reliable choice when you don't know the data's distribution shape in
> advance.

**Q9. How would you extend this project to handle much larger datasets?**
> Move from Pandas to something like Polars or Dask for out-of-core /
> parallel processing, add batching for the ingestion step, and move
> the AI audit to work off a *sample* of rows plus aggregate metadata
> rather than the full dataset, since the LLM call is the expensive,
> slow part.

**Q10. What was the hardest design decision in this project?**
> Deciding where to draw the line between "deterministic" and "AI"
> responsibilities. The guiding principle I used: anything that
> changes the actual data values must be deterministic and testable;
> anything that only explains, summarizes, or recommends can safely
> use an LLM, because a wrong recommendation is just ignored, but a
> wrong transformation corrupts the dataset.
