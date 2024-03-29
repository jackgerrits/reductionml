use crate::error::Result;

pub fn enforce_min_prob(
    uniform_epsilon: f32,
    consider_zero_valued_elements: bool,
    elements: &mut [(usize, f32)],
) -> Result<()> {
    if elements.len() == 0 {
        return Err(crate::error::Error::InvalidArgument(
            "elements.len() == 0".to_string(),
        ));
    }

    if uniform_epsilon == 0.0 {
        return Ok(());
    }

    if uniform_epsilon < 0.0 || uniform_epsilon > 1.0 {
        return Err(crate::error::Error::InvalidArgument(format!(
            "uniform_epsilon must be in [0, 1], but is {}",
            uniform_epsilon
        )));
    }

    let num_actions = elements.len();
    let support_size = if consider_zero_valued_elements {
        num_actions
    } else {
        num_actions - elements.iter().filter(|(_, p)| *p == 0.0).count()
    };

    if uniform_epsilon > 0.999 {
        elements.iter_mut().for_each(|(_, p)| {
            if consider_zero_valued_elements || *p > 0.0 {
                *p = 1.0 / support_size as f32;
            }
        });

        return Ok(());
    }

    let minimum_probability = uniform_epsilon / support_size as f32;
    let mut elements_copy = elements.to_vec();
    // Descending order. Args flipped
    elements_copy.sort_by(|(_, p1), (_, p2)| p2.partial_cmp(p1).unwrap());

    let mut idx = 0;
    let mut running_sum = 0.0;
    let mut rho_idx = 0;
    let mut rho_sum = elements_copy[0].1;

    for (_, prob) in elements_copy {
        if !consider_zero_valued_elements && prob == 0.0 {
            break;
        }
        running_sum += prob;
        if prob
            > ((support_size - idx - 1) as f32 * minimum_probability + running_sum - 1.0)
                / (idx as f32 + 1.0)
                + minimum_probability
        {
            rho_idx = idx;
            rho_sum = running_sum;
        }

        idx += 1;
    }

    let tau = ((support_size as f32 - rho_idx as f32 - 1.0) * minimum_probability + rho_sum - 1.0)
        / (rho_idx as f32 + 1.0);
    elements.iter_mut().for_each(|(_, p)| {
        if consider_zero_valued_elements || *p > 0.0 {
            *p = (*p - tau).max(minimum_probability);
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::enforce_min_prob;

    #[test]
    fn test_enforce_minimum_probability() {
        let mut input = vec![(0, 1.0), (0, 0.0), (0, 0.0)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_no_zeros() {
        let mut input = vec![(0, 0.9), (0, 0.1), (0, 0.0)];
        enforce_min_prob(0.6, false, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.7, 0.3, 0.0].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_all_zeros_and_dont_consider() {
        let mut input = vec![(0, 0.0), (0, 0.0), (0, 0.0)];
        enforce_min_prob(0.6, false, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.0, 0.0, 0.0].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_all_zeros_and_consider() {
        let mut input = vec![(0, 0.0), (0, 0.0), (0, 0.0)];
        enforce_min_prob(0.6, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(
            just_probs.as_slice(),
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0].as_slice()
        );
    }

    #[test]
    fn test_enforce_minimum_probability_equal_to_amt() {
        let mut input = vec![(0, 0.0), (0, 2.0 / 3.0), (0, 1.0 / 3.0)];
        enforce_min_prob(1.0, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(
            just_probs.as_slice(),
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0].as_slice()
        );
    }
    #[test]
    fn test_enforce_minimum_probability_uniform() {
        let mut input = vec![(0, 0.9), (0, 0.1), (0, 0.0), (0, 0.0)];
        enforce_min_prob(1.0, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(
            just_probs.as_slice(),
            vec![0.25, 0.25, 0.25, 0.25].as_slice()
        );
    }

    #[test]
    #[should_panic]
    fn test_enforce_minimum_probability_bad_range() {
        enforce_min_prob(1.0, false, &mut vec![]).unwrap();
    }
    #[test]
    fn test_enforce_minimum_probability_uniform1() {
        let mut input = vec![(0, 0.9), (0, 0.1), (0, 0.0)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_uniform2() {
        let mut input = vec![(0, 0.8), (0, 0.1), (0, 0.1)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_uniform_unsorted() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.1), (0, 0.8), (0, 0.1)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.1, 0.8, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_bug_incl_zero() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.89), (0, 0.11), (0, 0.0)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_zero_epsilon_dont_consider() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.89), (0, 0.11), (0, 0.0)];
        enforce_min_prob(0.0, false, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.89, 0.11, 0.0].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_zero_epsilon_consider() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.89), (0, 0.11), (0, 0.0)];
        enforce_min_prob(0.0, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.89, 0.11, 0.0].as_slice());
    }
}
