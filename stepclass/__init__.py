try:
    import lsst.log
    lsstLog = lsst.log.Log()
    lsstLog.setLevel(lsst.log.ERROR)
    LSST_INSTALLED = True
except ImportError:
    LSST_INSTALLED = False
