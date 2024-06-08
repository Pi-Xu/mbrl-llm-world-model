# main.py

# This will trigger the algorithm registration of SheepRL
from algos import dreamer_v3  # noqa: F401

if __name__ == "__main__":
    # This must be imported after the algorithm registration, otherwise SheepRL
    # will not be able to find the new algorithm given the name specified
    # in the `algo.name` field of the `./configs/algo/dreamer_v3.yaml` config file
    from sheeprl.cli import run

    run()