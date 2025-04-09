FROM sinapsis-nvidia:base

COPY . /app/
WORKDIR /app
RUN uv sync --frozen

RUN uv pip install sinapsis-framework-converter[all] --extra-index-url https://pypi.sinapsis.tech
RUN uv pip install tensorflow[and-cuda]==2.18.0

