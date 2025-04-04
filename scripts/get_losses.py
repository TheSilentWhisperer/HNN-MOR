if __name__ == "__main__":
    
    from utils.plot import plot_losses
    import os
    import json
    
    config = json.load(open("scripts/config.json"))
    run_id = config["RUN_ID"]
    run_info_path = os.path.join("runs", run_id, "run_info.json")
    config_path = os.path.join("runs", run_id, "config.json")
    run_info = json.load(open(run_info_path))
    config = json.load(open(config_path))
    plot_losses(run_info, config)
    print("Plot saved at runs/{}/loss_plot.png".format(run_id))
