import numpy as np
class SpectralSelection:
    SUPPORTED={"Nearest","Exact","Window"}
    @staticmethod
    def nearest_index(x_axis,center):
        x=np.asarray(x_axis,float)
        if x.ndim!=1 or not x.size or not np.all(np.isfinite(x)):raise ValueError("Invalid spectral axis")
        return int(np.argmin(abs(x-float(center))))
    @staticmethod
    def nearest_value(x_axis,center):
        return float(x_axis[SpectralSelection.nearest_index(x_axis,center)])
    @staticmethod
    def select_indices(x_axis,center,strategy="Nearest",tolerance=0.):
        x=np.asarray(x_axis,float)
        strategy=strategy.value if hasattr(strategy,"value") else str(strategy)
        nearest=SpectralSelection.nearest_index(x,center)
        if strategy=="Nearest":return np.array([nearest],dtype=np.int64)
        if strategy=="Exact":
            eps=8*np.finfo(float).eps*max(1.,float(np.max(abs(x))),abs(float(center)))
            ids=np.flatnonzero(np.isclose(x,float(center),rtol=0,atol=eps))
        elif strategy=="Window":
            if tolerance<0:raise ValueError("tolerance must be nonnegative")
            ids=np.flatnonzero(abs(x-float(center))<=float(tolerance))
        else:raise ValueError(f"Unknown selection {strategy}")
        if not ids.size:raise ValueError(f"No channel at {center} using {strategy}")
        return ids.astype(np.int64)
