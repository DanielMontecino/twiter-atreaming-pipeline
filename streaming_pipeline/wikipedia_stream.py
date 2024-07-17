import logging
from typing import List

from bytewax.inputs import DynamicInput, StatelessSource
from datasets import load_dataset

# Creating an object
logger = logging.getLogger()


class WikipediaArticlesStreamInput(DynamicInput):
    """Input class to stream Wikipedia articles data
    
    Args:
        title_prefixes: str: The prefix of the title to filter articles
    """

    def __init__(self, title_prefixes):
        self._title_prefixes = title_prefixes

    def build(self, worker_index, worker_count):
        """
        Distributes the title_prefixes to the workers. If parallelized,
        workers will filter articles based on the title_prefixes.

        Args:
            worker_index (int): The index of the current worker.
            worker_count (int): The total number of workers.

        Returns:
            WikipediaArticlesStreamSource: An instance of the WikipediaArticlesStreamSource class
            with the worker's allocated tickers.
        """

        if self._title_prefixes:
            prods_per_worker = int(len(self._title_prefixes) / worker_count)
            worker_prefixes = self._title_prefixes[
                int(worker_index * prods_per_worker) : int(
                    worker_index * prods_per_worker + prods_per_worker
                )
            ]
        else:
            worker_prefixes = None

        return WikipediaArticlesStreamSource(title_prefixes=worker_prefixes)


class WikipediaArticlesStreamSource(StatelessSource):
    """
    A source for streaming wikipedia articles.

    Args:
        title_prefixes (List[str]): A list of title prefixes to filter articles.

    Attributes:
        _wikipedia_dataset (datasets.dataset_dict.IterableDatasetDict): An instance of Wikipedi dataset.
    """

    def __init__(self, title_prefixes: List[str]):
        """
        Initializes the WikipediaArticlesStreamSource object.

        Args:
            title_prefixes (List[str]): A list of title prefixes to filter articles.
        """
        self._wikipedia_dataset = load_dataset("wikipedia", "20220301.simple", streaming=True)
        if title_prefixes:
            self._wikipedia_dataset.filter(lambda x: any(x["title"].startswith(prefix) for prefix in title_prefixes))
        self._wikipedia_dataset = self._wikipedia_dataset["train"].__iter__()
        print("WikipediaArticlesStreamSource initialized")
        self.counter = 0

    def next(self):
        """
        Returns the next wikipedia article item from dataset.

        Returns:
            dict: A dictionary containing the wikipedia article item data.
        """
        print(f"Getting next item {self.counter}")
        self.counter += 1
        return [next(self._wikipedia_dataset)]

