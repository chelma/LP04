# LP04

This package contains some prototype code to learn more about creating summarized extracts of reference documentation that be provide grounded knowledge for LLM inference.  It uses Beautiful Soup and LangChain to extract the text from a web page, convert it to markdown, and summarize/compress the contents.  Particularly large web pages are likely to break the tool, as the Bedrock inference API has a maximum output of 4096 tokens.

### Running the code

#### Locally
To run the code locally, use a Python virtual environment.  You'll need AWS Credentials in your AWS Keyring, permissions to invoke Bedrock, and to have onboarded your account to use Claude 3.5 Sonnet.

```
# Start in the repo root

python3 -m venv venv
source venv/bin/activate

(cd lp04 && pipenv sync --dev)
python3 ./lp04/gen_summary.py --url "https://opensearch.org/docs/2.17/api-reference/index-apis/create-index/" --output "../generated_summary.md"
```

### Dependencies
`pipenv` is used to managed dependencies within the project.  The `Pipefile` and `Pipefile.lock` handle the local environment.  You can add dependencies like so:

```
pipenv install boto3
```

This updates the `Pipfile`/`Pipfile.lock` with the new dependency.  To create a local copy of the dependencies, such as for bundling a distribution, you can use pip like so:

```
pipenv requirements > requirements.txt
python3 -m pip install -r requirements.txt -t ./package --upgrade

zip -r9 lp04.zip tools/ package/
```