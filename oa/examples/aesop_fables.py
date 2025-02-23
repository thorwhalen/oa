"""Making a children book containing rhyming stories of aesop fables,
illustrated using different styles of images, taking art movements and
famous artists as styles."""

from typing import Mapping, MutableMapping
import os
import io
from itertools import cycle

import pandas as pd
import requests
from tabled import get_tables_from_url
from dol import Files, TextFiles, wrap_kvs


# --------------------------------------------------------------------------------------
# Stores
# Here, we'll use `dol` to make some "stores" -- that is, a `MutableMapping` facade to
# where we'll store stuff (our fables text, rhyming stories, illustration urls,
# images...).
# We'll store things in local files here, but we can change this to use S3, DBs, etc.
# simply by changing the backend of the facade.
def rm_extension(ext):
    """Make a key transformer that removes the given extension from keys"""
    if not ext.startswith("."):
        ext = "." + ext
    return wrap_kvs(id_of_key=lambda x: x + ext, key_of_id=lambda x: x[: -len(ext)])


Texts = rm_extension("txt")(TextFiles)
Images = rm_extension("jpg")(Files)
Htmls = rm_extension("html")(TextFiles)

# --------------------------------------------------------------------------------------


root_url = "https://aesopfables.com/"
url = root_url + "aesopsel.html"


def url_to_bytes(url: str) -> bytes:
    return requests.get(url).content


def _clean_up_fable_table(t):
    t.columns = ["fable", "moral"]
    t["moral"] = t["moral"].map(lambda x: x[0].strip())
    t["moral"] = t["moral"].map(lambda x: x[1:] if x.startswith(".") else x)
    t["title"], t["rel_url"] = zip(*t["fable"])
    t["url"] = t["rel_url"].map(lambda x: root_url + x)
    del t["fable"]
    return t


def get_fable_table(files):
    if "fables.csv" not in files:
        df = get_tables_from_url(url, extract_links="all")[0]
        df = _clean_up_fable_table(df)
        files["fables.csv"] = df.to_csv(index=False).encode()
    return pd.read_csv(io.BytesIO(files["fables.csv"]))


def get_title_and_urls(fable_table):
    return dict(zip(fable_table["title"], fable_table["url"]))


def get_original_story(url):
    import requests
    from bs4 import BeautifulSoup

    r = requests.get(url)
    soup = BeautifulSoup(r.content)
    return soup.find("pre").text.strip()


# TODO: Make decorators for launch_iteration, print_progress, and overwrite concerns.


def get_original_stories(
    title_and_urls: Mapping,
    original_stories: MutableMapping,
    *,
    launch_iteration=True,
    print_progress=True,
    overwrite=False,
):
    """Extract and store the text of each fable in the fable table"""
    n = len(title_and_urls)

    def run_process():
        for i, (title, url) in enumerate(title_and_urls.items(), 1):
            if print_progress:
                print(f"{i}/{n}: {title}")
            if overwrite or title not in original_stories:
                original_story = get_original_story(url)
                original_stories[title] = original_story

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


import oa.examples.illustrate_stories as ii


def get_rhyming_stories(
    original_stories: Mapping,
    *,
    rhyming_stories: MutableMapping,
    launch_iteration=True,
    overwrite=False,
    print_progress=True,
    **kwargs,
):
    n = len(original_stories)

    def run_process():
        for i, (title, original_story) in enumerate(original_stories.items(), 1):
            if print_progress:
                print(f"{i}/{n}: {title}")
            if overwrite or title not in rhyming_stories:
                original_story = original_stories[title]
                rhyming_story = ii.make_it_rhyming(original_story, **kwargs)
                rhyming_stories[title] = rhyming_story

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


def get_image_descriptions(
    stories: Mapping,
    *,
    image_descriptions: MutableMapping,
    image_styles=("children's book drawing",),
    launch_iteration=True,
    overwrite=False,
    print_progress=True,
    **kwargs,
):
    _image_styles = cycle(image_styles)
    n = len(stories)

    def run_process():
        for i, (title, story) in enumerate(stories.items(), 1):
            image_style = next(_image_styles)
            if print_progress:
                print(f"{i}/{n}: {title=}, {image_style=}")

            if overwrite or title not in image_descriptions:
                image_description = ii.get_image_description(
                    story, image_style, **kwargs
                )
                image_descriptions[title] = image_description
            yield

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


def get_images(
    image_descriptions: Mapping,
    images: MutableMapping,
    *,
    image_urls: Mapping = None,
    image_styles=("children's book drawing",),
    launch_iteration=True,
    overwrite=False,
    print_progress=True,
    **kwargs,
):
    n = len(image_descriptions)
    _image_styles = cycle(image_styles)

    def run_process():
        for i, (title, image_description) in enumerate(image_descriptions.items(), 1):
            image_style = next(_image_styles)
            if overwrite or title not in images:
                try:
                    if print_progress:
                        print(f"{i}/{n}: {title}")
                    image_url = ii.get_image_url(
                        image_description, image_style, **kwargs
                    )
                    if image_urls is not None:
                        image_urls[title] = image_url
                    images[title] = url_to_bytes(image_url)
                except Exception as e:
                    print(
                        f"The description or url for {title} lead to the error {e}. "
                        f"Description:\n\n{image_description}\n\n\n"
                    )
            yield

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


def store_stats(*, original_stories, rhyming_stories, image_descriptions, image_urls):
    print(f"{len(original_stories)=}")
    print(f"{len(rhyming_stories)=}")
    print(f"{len(image_descriptions)=}")
    print(f"{len(image_urls)=}")
    print("")
    missing_rhyming_stories = set(original_stories) - set(rhyming_stories)
    missing_descriptions = set(rhyming_stories) - set(image_descriptions)
    missing_urls = set(image_descriptions) - set(image_urls)
    print(f"{len(missing_rhyming_stories)=}")
    print(f"{len(missing_descriptions)=}")
    print(f"{len(missing_urls)=}")


def mk_pages_store(*, rhyming_stories, image_urls, ipython_display=False):
    from dol import wrap_kvs, add_ipython_key_completions
    from dol.sources import FanoutReader

    fanout_store = add_ipython_key_completions(
        FanoutReader(
            {
                "rhyming_stories": rhyming_stories,
                "image_urls": image_urls,
            },
            keys=image_urls,  # take keys from image_urls
        )
    )

    s = wrap_kvs(
        fanout_store,
        obj_of_data=lambda x: ii.aggregate_story_and_image(
            image_url=x["image_urls"], story_text=x["rhyming_stories"]
        ),
    )
    if ipython_display:
        from IPython.display import HTML

        s = wrap_kvs(s, obj_of_data=HTML)
    return s


# --------------------------------------------------------------------------------------
# Resources


def get_top100_artists():
    import requests
    from operator import attrgetter, itemgetter, methodcaller
    from bs4 import BeautifulSoup
    from dol import Pipe

    get_soup = Pipe(requests.get, attrgetter("content"), BeautifulSoup)

    soup = get_soup("https://www.art-prints-on-demand.com/a/artists-painters.html")
    soup2 = get_soup(
        "https://www.art-prints-on-demand.com/a/artists-painters.html&mpos=999&ALL_ABC=1"
    )
    # extract artists
    get_title = Pipe(methodcaller("find", "a"), itemgetter("title"))
    t1 = list(
        map(get_title, soup.find_all("div", {"class": "kk_category_pic"}))
    )  # 30 top
    t2 = list(
        map(get_title, soup2.find_all("div", {"class": "kk_category_pic"}))
    )  # 100 top
    # merge both lists leaving the top 30 at the top, to favor them
    t = t1 + t2
    top_artists = [x for i, x in enumerate(t) if x not in (t)[:i]]
    return top_artists


# Note: Obtained from get_top100_artists()
top100_artists = [
    "Claude Monet",
    "Gustav Klimt",
    "Vincent van Gogh",
    "Paul Klee",
    "Wassily Kandinsky",
    "Franz Marc",
    "Caspar David Friedrich",
    "August Macke",
    "Egon Schiele",
    "Pierre-Auguste Renoir",
    "William Turner",
    "Leonardo da Vinci",
    "Johannes Vermeer",
    "Albrecht Dürer",
    "Carl Spitzweg",
    "Alphonse Mucha",
    "Catrin Welz-Stein",
    "Max Liebermann",
    "Paul Cézanne",
    "Rembrandt van Rijn",
    "Paul Gauguin",
    "(Raphael) Raffaello Sanzio",
    "Amadeo Modigliani",
    "Sandro Botticelli",
    "Edvard Munch",
    "Pierre Joseph Redouté",
    "Michelangelo Caravaggio",
    "Ernst Ludwig Kirchner",
    "Piet Mondrian",
    "Pablo Picasso",
    "Katsushika Hokusai",
    "Hieronymus Bosch",
    "Timothy  Easton",
    "Paula Modersohn-Becker",
    "Edgar Degas",
    "Michelangelo (Buonarroti)",
    "Salvador Dali",
    "Gustave Caillebotte",
    "Pieter Brueghel the Elder",
    "Ferdinand Hodler",
    "Joan Miró",
    "John William Waterhouse",
    "Peter Severin Kroyer",
    "Peter Paul Rubens",
    "Peter  Graham",
    "Henri de Toulouse-Lautrec",
    "Camille Pissarro",
    "Edouard Manet",
    "Joaquin Sorolla",
    "Sara Catena",
    "Henri Julien-Félix Rousseau",
    "Gustave Courbet",
    "Jack Vettriano",
    "Felix Vallotton",
    "All catalogs",
    "Arnold Böcklin",
    "Alexej von Jawlensky",
    "Kazimir Severinovich Malewitsch",
    "Odilon Redon",
    "Jean-Étienne Liotard",
    "Giovanni Segantini",
    "Azure",
    "Oskar Schlemmer",
    "Carl Larsson",
    "Francisco José de Goya",
    "Artist Artist",
    "François Boucher",
    "Mark Rothko",
    "Susett Heise",
    "Alfred Sisley",
    "Giovanni Antonio Canal (Canaletto)",
    "Jean-François Millet",
    "Giuseppe Arcimboldo",
    "Iwan Konstantinowitsch Aiwasowski",
    "Catherine  Abel",
    "Edward Hopper",
    "Mark  Adlington",
    "Jean Honoré Fragonard",
    "Lucy Willis",
    "Jacques Louis David",
    "Pavel van Golod",
    "M.c. Escher",
    "Pierre Bonnard",
    "Ferdinand Victor Eugène Delacroix",
    "Carel Fabritius",
    "Franz von Stuck",
    "John Constable",
    "László Moholy-Nagy",
    "Lincoln  Seligman",
    "William Adolphe Bouguereau",
    "Adolph Friedrich Erdmann von Menzel",
    "Petra Schüßler",
    "Pompei, wall painting",
    "Unbekannter Künstler",
    "Ando oder Utagawa Hiroshige",
    "Marc Chagall",
    "Zita Rauschgold",
    "William  Ireland",
    "Bernardo Bellotto",
    "Hermann Angeli",
]

# Note: Reference: https://magazine.artland.com/art-movements-and-styles/
art_movements = """Abstract Expressionism
Art Deco
Art Nouveau
Avant-garde
Baroque
Bauhaus
Classicism
CoBrA
Color Field Painting
Conceptual Art
Constructivism
Cubism
Dada / Dadaism
Digital Art
Expressionism
Fauvism
Futurism
Harlem Renaissance
Impressionism
Installation Art
Land Art
Minimalism
Neo-Impressionism
Neoclassicism
Neon Art
Op Art
Performance Art
Pop Art
Post-Impressionism
Precisionism
Rococo
Street Art
Surrealism
Suprematism
Symbolism
Zero Group""".splitlines()

styles_of_art = art_movements + top100_artists
