use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

use crate::loss_function::LossFunctionImpl;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema, DefaultFromSerde)]
pub struct LogisticLoss {
    #[serde(default = "default_min")]
    min_label: f32,
    #[serde(default = "default_max")]
    max_label: f32,
}

fn default_min() -> f32 {
    -1.
}

fn default_max() -> f32 {
    1.
}
fn get_loss_sub(prediction: f32, label: f32) -> f32 {
    debug_assert!(label == -1. || label == 1.);
    return (1. + (-label * prediction).exp()).ln();
}

// Based on implementation in VowpalWabbit
fn wexpmx(x: f32) -> f32 {
    /* This piece of code is approximating W(exp(x))-x.
     * W is the   : W(z)*exp(W(z))=z.
     * The absolute error of this approximation is less than 9e-5.
     * Faster/better approximations can be substituted here.
     */
    let x: f64 = x as f64;
    let w: f64 = if x >= 1. {
        0.86 * x + 0.01
    } else {
        (-0.65 + 0.8 * x).exp()
    }; // initial guess
    let r = if x >= 1. {
        x - w.ln() - w
    } else {
        0.2 * x + 0.65 - w
    }; // residual
    let t = 1. + w;
    let u = 2. * t * (t + 2. * r / 3.); // magic
    (w * (1. + r / t * (u - r) / (u - 2. * r)) - x) as f32 // more magic
}

fn get_update_sub(prediction: f32, label: f32, update_scale: f32, pred_per_update: f32) -> f32 {
    let d = (label * prediction).exp();
    if update_scale * pred_per_update < 1e-6 {
        /* As with squared loss, for small eta_t we replace the update
         * with its first order Taylor expansion to avoid numerical problems
         */
        return label * update_scale / (1. + d);
    }
    let x = update_scale * pred_per_update + label * prediction + d;
    let w = wexpmx(x);
    -(label * w + prediction) / pred_per_update
}

fn get_unsafe_update_sub(prediction: f32, label: f32, update_scale: f32) -> f32 {
    let d = (label * prediction).exp();
    return label * update_scale / (1.0 + d);
}

fn first_derivative_sub(prediction: f32, label: f32) -> f32 {
    let v = -label / (1.0 + (label * prediction).exp());
    return v;
}

fn second_derivative_sub(prediction: f32, label: f32) -> f32 {
    let p = 1.0 / (1.0 + (label * prediction).exp());
    return p * (1.0 - p);
}

impl LogisticLoss {
    pub fn new(min_label: f32, max_label: f32) -> LogisticLoss {
        LogisticLoss {
            min_label,
            max_label,
        }
    }
}

impl LossFunctionImpl for LogisticLoss {
    fn get_loss(&self, _min_label: f32, _max_label: f32, prediction: f32, label: f32) -> f32 {
        debug_assert!(label >= self.min_label && label <= self.max_label);
        let std_label = (label - self.min_label) / (self.max_label - self.min_label);
        return std_label * get_loss_sub(prediction, 1.0)
            + (1.0 - std_label) * get_loss_sub(prediction, -1.0);
    }

    fn get_update(
        &self,
        prediction: f32,
        label: f32,
        update_scale: f32,
        pred_per_update: f32,
    ) -> f32 {
        let std_label = (label - self.min_label) / (self.max_label - self.min_label);
        return std_label * get_update_sub(prediction, 1.0, update_scale, pred_per_update)
            + (1.0 - std_label) * get_update_sub(prediction, -1.0, update_scale, pred_per_update);
    }

    fn get_unsafe_update(&self, prediction: f32, label: f32, update_scale: f32) -> f32 {
        let std_label = (label - self.min_label) / (self.max_label - self.min_label);
        return std_label * get_unsafe_update_sub(prediction, 1.0, update_scale)
            + (1.0 - std_label) * get_unsafe_update_sub(prediction, -1.0, update_scale);
    }

    fn get_square_grad(&self, prediction: f32, label: f32) -> f32 {
        let d = self.first_derivative(0.0, 0.0, prediction, label);
        return d * d;
    }

    fn first_derivative(
        &self,
        _min_label: f32,
        _max_label: f32,
        prediction: f32,
        label: f32,
    ) -> f32 {
        let std_label = (label - self.min_label) / (self.max_label - self.min_label);
        return std_label * first_derivative_sub(prediction, 1.0)
            + (1.0 - std_label) * first_derivative_sub(prediction, -1.0);
    }

    fn second_derivative(
        &self,
        _min_label: f32,
        _max_label: f32,
        prediction: f32,
        label: f32,
    ) -> f32 {
        let std_label = (label - self.min_label) / (self.max_label - self.min_label);
        return std_label * second_derivative_sub(prediction, 1.0)
            + (1.0 - std_label) * second_derivative_sub(prediction, -1.0);
    }
}
