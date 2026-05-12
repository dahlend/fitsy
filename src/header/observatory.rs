//! Observatory location keywords (WCS Paper IV Sec.3.1.4).

use crate::header::Header;

// WGS84 reference ellipsoid. WCS Paper IV Sec.3.1.4 cites the IAU 1976
// ellipsoid (a=6378140, 1/f=298.2577) but also states that nanosecond
// precision requires a post-1984 geodetic frame; WGS84 is the standard
// used by GPS receivers and modern FITS writers.
const ELLIPSOID_A: f64 = 6_378_137.0; // semi-major axis, m
const ELLIPSOID_INV_F: f64 = 298.257_223_563; // inverse flattening

/// Observatory location as ITRS Cartesian coordinates (m), geocentric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObsGeo {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ObsGeo {
    /// Convert to geodetic coordinates on the WGS84 ellipsoid (Bowring 1985).
    #[must_use]
    pub fn to_geodetic(&self) -> ObsGeodetic {
        cartesian_to_geodetic(self)
    }
}

/// Observatory location as geodetic coordinates on the WGS84 ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObsGeodetic {
    /// Geodetic latitude in degrees, North positive.
    pub lat: f64,
    /// Geodetic longitude in degrees, East positive.
    pub lon: f64,
    /// Height above the WGS84 ellipsoid in meters.
    pub alt: f64,
}

impl ObsGeodetic {
    /// Convert to ITRS Cartesian coordinates using the WGS84 ellipsoid
    /// (WCS Paper IV Sec.3.1.4).
    #[must_use]
    pub fn to_cartesian(&self) -> ObsGeo {
        geodetic_to_cartesian(self)
    }
}

/// Convert geodetic to ITRS Cartesian (WCS Paper IV Sec.3.1.4 equations).
fn geodetic_to_cartesian(g: &ObsGeodetic) -> ObsGeo {
    let f = 1.0 / ELLIPSOID_INV_F;
    let e2 = 2.0 * f - f * f;
    let lat = g.lat.to_radians();
    let lon = g.lon.to_radians();
    let sin_b = lat.sin();
    let cos_b = lat.cos();
    // N(B): radius of curvature in the prime vertical.
    let n = ELLIPSOID_A / (1.0 - e2 * sin_b * sin_b).sqrt();
    ObsGeo {
        x: (n + g.alt) * cos_b * lon.cos(),
        y: (n + g.alt) * cos_b * lon.sin(),
        z: (n * (1.0 - e2) + g.alt) * sin_b,
    }
}

/// Convert ITRS Cartesian to geodetic using the Bowring (1985) closed-form
/// approximation, accurate to sub-millimetre for Earth-surface coordinates.
fn cartesian_to_geodetic(c: &ObsGeo) -> ObsGeodetic {
    let f = 1.0 / ELLIPSOID_INV_F;
    let e2 = 2.0 * f - f * f;
    let b = ELLIPSOID_A * (1.0 - f); // semi-minor axis
    let ep2 = (ELLIPSOID_A * ELLIPSOID_A - b * b) / (b * b); // second eccentricity squared

    let lon = c.y.atan2(c.x);
    let p = c.x.hypot(c.y); // distance from polar axis

    // Bowring's parametric (reduced) latitude estimate.
    let theta = (c.z * ELLIPSOID_A).atan2(p * b);
    let sin_t = theta.sin();
    let cos_t = theta.cos();

    let lat = (c.z + ep2 * b * sin_t.powi(3)).atan2(p - e2 * ELLIPSOID_A * cos_t.powi(3));
    let sin_b = lat.sin();
    let cos_b = lat.cos();

    let n = ELLIPSOID_A / (1.0 - e2 * sin_b * sin_b).sqrt();
    // Choose numerically stable formula based on latitude.
    let alt = if cos_b.abs() > sin_b.abs() {
        p / cos_b - n
    } else {
        c.z / sin_b - n * (1.0 - e2)
    };

    ObsGeodetic {
        lat: lat.to_degrees(),
        lon: lon.to_degrees(),
        alt,
    }
}

impl Header {
    /// Reads `OBSGEO-X/Y/Z` from keywords, without any fallback.
    fn read_cartesian_raw(&self) -> Option<ObsGeo> {
        Some(ObsGeo {
            x: self.optional_real("OBSGEO-X")?,
            y: self.optional_real("OBSGEO-Y")?,
            z: self.optional_real("OBSGEO-Z")?,
        })
    }

    /// Reads geodetic keywords without any fallback.
    fn read_geodetic_raw(&self) -> Option<ObsGeodetic> {
        let lat = self
            .optional_real("OBSGEO-B")
            .or_else(|| self.optional_real("LAT-OBS"))
            .or_else(|| self.optional_real("OBS-LAT"))
            .or_else(|| self.optional_real("OBSLAT"))
            .or_else(|| self.optional_real("SITELAT"))?;
        let lon = self
            .optional_real("OBSGEO-L")
            .or_else(|| self.optional_real("LONG-OBS"))
            .or_else(|| self.optional_real("OBS-LONG"))
            .or_else(|| self.optional_real("OBSLONG"))
            .or_else(|| self.optional_real("SITELONG"))?;
        let alt = self
            .optional_real("OBSGEO-H")
            .or_else(|| self.optional_real("OBS-ELEV"))
            .or_else(|| self.optional_real("OBSELEV"))
            .or_else(|| self.optional_real("ALT-OBS"))
            .or_else(|| self.optional_real("SITEELEV"))
            .unwrap_or(0.0);
        Some(ObsGeodetic { lat, lon, alt })
    }

    /// Observatory location as ECEF/ITRS Cartesian coordinates (m).
    ///
    /// Reads `OBSGEO-X/Y/Z` directly; falls back to geodetic keywords
    /// converted via the WGS84 ellipsoid. Returns `None` if no location
    /// keywords are present.
    #[must_use]
    pub fn obs_ecef(&self) -> Option<ObsGeo> {
        self.read_cartesian_raw()
            .or_else(|| Some(self.read_geodetic_raw()?.to_cartesian()))
    }

    /// Observatory geodetic coordinates on the WGS84 ellipsoid.
    ///
    /// Reads `OBSGEO-B/L/H` and non-standard variants first (WCS Paper IV
    /// Sec.3.1.4); falls back to `OBSGEO-X/Y/Z` converted via the Bowring
    /// (1985) inverse. Altitude defaults to `0.0` m when only lat/lon
    /// keywords are present. Returns `None` if no location keywords exist.
    #[must_use]
    pub fn obs_geodetic(&self) -> Option<ObsGeodetic> {
        self.read_geodetic_raw()
            .or_else(|| Some(self.read_cartesian_raw()?.to_geodetic()))
    }

    /// Orbit ephemeris file (`OBSORBIT`): URI, URL, or name.
    #[must_use]
    pub fn obs_orbit(&self) -> Option<String> {
        self.optional_string("OBSORBIT").map(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(cards: &[(&str, f64)]) -> Header {
        let mut h = Header::empty();
        for (k, v) in cards {
            h.push(*k, *v, None).unwrap();
        }
        h
    }

    #[test]
    fn obsgeo_reads_cartesian_directly() {
        let h = make_header(&[
            ("OBSGEO-X", 100.0),
            ("OBSGEO-Y", 200.0),
            ("OBSGEO-Z", 300.0),
        ]);
        assert_eq!(
            h.obs_ecef().unwrap(),
            ObsGeo {
                x: 100.0,
                y: 200.0,
                z: 300.0
            }
        );
    }

    #[test]
    fn obsgeo_converts_from_geodetic() {
        // Equator, prime meridian, sea level -> X=a, Y=0, Z=0.
        let h = make_header(&[("OBSGEO-B", 0.0), ("OBSGEO-L", 0.0), ("OBSGEO-H", 0.0)]);
        let g = h.obs_ecef().unwrap();
        assert!((g.x - ELLIPSOID_A).abs() < 1.0, "X={}", g.x);
        assert!(g.y.abs() < 1e-6, "Y={}", g.y);
        assert!(g.z.abs() < 1e-6, "Z={}", g.z);
    }

    #[test]
    fn obsgeo_cartesian_preferred_over_geodetic() {
        let mut h = make_header(&[
            ("OBSGEO-X", 1.0),
            ("OBSGEO-Y", 2.0),
            ("OBSGEO-Z", 3.0),
            ("OBSGEO-B", 45.0),
            ("OBSGEO-L", 10.0),
        ]);
        h.push("OBSGEO-H", 0.0_f64, None).unwrap();
        // X/Y/Z must win.
        assert_eq!(
            h.obs_ecef().unwrap(),
            ObsGeo {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
    }

    #[test]
    fn obs_geodetic_reads_geodetic_directly() {
        let h = make_header(&[
            ("OBSGEO-B", -30.0),
            ("OBSGEO-L", 150.0),
            ("OBSGEO-H", 1000.0),
        ]);
        let g = h.obs_geodetic().unwrap();
        assert_eq!((g.lat, g.lon, g.alt), (-30.0, 150.0, 1000.0));
    }

    #[test]
    fn obs_geodetic_falls_back_to_cartesian() {
        // Only X/Y/Z present: obs_geodetic should convert back to lat/lon/alt.
        // Use equator, prime meridian so the expected values are trivial.
        let h = make_header(&[
            ("OBSGEO-X", ELLIPSOID_A),
            ("OBSGEO-Y", 0.0),
            ("OBSGEO-Z", 0.0),
        ]);
        let g = h.obs_geodetic().unwrap();
        assert!(g.lat.abs() < 1e-6, "lat={}", g.lat);
        assert!(g.lon.abs() < 1e-6, "lon={}", g.lon);
        assert!(g.alt.abs() < 1e-3, "alt={}", g.alt);
    }

    #[test]
    fn obs_geodetic_geodetic_preferred_over_cartesian() {
        let mut h = make_header(&[
            ("OBSGEO-B", -30.0),
            ("OBSGEO-L", 150.0),
            ("OBSGEO-H", 500.0),
            ("OBSGEO-X", 1.0),
            ("OBSGEO-Y", 2.0),
        ]);
        h.push("OBSGEO-Z", 3.0_f64, None).unwrap();
        // B/L/H must win.
        let g = h.obs_geodetic().unwrap();
        assert_eq!((g.lat, g.lon, g.alt), (-30.0, 150.0, 500.0));
    }

    #[test]
    fn obs_geodetic_fallback_keywords() {
        let mut h = Header::empty();
        h.push("SITELAT", -30.0_f64, None).unwrap();
        h.push("SITELONG", 150.0_f64, None).unwrap();
        h.push("SITEELEV", 1000.0_f64, None).unwrap();
        let g = h.obs_geodetic().unwrap();
        assert_eq!((g.lat, g.lon, g.alt), (-30.0, 150.0, 1000.0));
    }

    #[test]
    fn obs_geodetic_altitude_defaults_to_zero() {
        let h = make_header(&[("OBSGEO-B", 10.0), ("OBSGEO-L", 20.0)]);
        assert_eq!(h.obs_geodetic().unwrap().alt, 0.0);
    }

    #[test]
    fn round_trip_geodetic_cartesian() {
        // VLT Paranal: lat=-24.6157, lon=-70.3976, alt=2635 m.
        let orig = ObsGeodetic {
            lat: -24.6157,
            lon: -70.3976,
            alt: 2635.0,
        };
        let cart = orig.to_cartesian();
        let back = cart.to_geodetic();
        assert!(
            (back.lat - orig.lat).abs() < 1e-6,
            "lat diff={}",
            back.lat - orig.lat
        );
        assert!(
            (back.lon - orig.lon).abs() < 1e-6,
            "lon diff={}",
            back.lon - orig.lon
        );
        assert!(
            (back.alt - orig.alt).abs() < 1e-3,
            "alt diff={}",
            back.alt - orig.alt
        );
    }

    #[test]
    fn geodetic_north_pole() {
        // North pole, sea level -> X~0, Y~0, Z~b.
        let c = ObsGeodetic {
            lat: 90.0,
            lon: 0.0,
            alt: 0.0,
        }
        .to_cartesian();
        let f = 1.0 / ELLIPSOID_INV_F;
        let b = ELLIPSOID_A * (1.0 - f);
        assert!(c.x.abs() < 1.0, "X={}", c.x);
        assert!(c.y.abs() < 1.0, "Y={}", c.y);
        assert!((c.z - b).abs() < 1.0, "Z={} b={}", c.z, b);
    }
}
