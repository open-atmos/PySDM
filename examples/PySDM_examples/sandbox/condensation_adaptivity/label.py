def label(settings):
    lbl = str(
        {
            k.replace("condensation_", ""): (
                f"{v:.1e}"
                if isinstance(v, float)
                else str(v).zfill(2) if isinstance(v, int) else v
            )
            for k, v in settings.items()
        }
    )
    return (
        lbl.replace("{", "")
        .replace("}", "")
        .replace("'", "")
        .replace("True", "T")
        .replace("False", "F")
        .replace("_thd", "$_{th}$")
        .replace("e-0", "e-")
    )
