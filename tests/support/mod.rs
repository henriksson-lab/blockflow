#[allow(unused_macros)]
macro_rules! refuses {
    ($context:expr => $result:expr, $($needle:expr),+ $(,)?) => {{
        let message = match $result {
            Ok(_) => panic!(
                "{}: {} must be refused, but it succeeded",
                $context,
                stringify!($result)
            ),
            Err(error) => error.to_string(),
        };
        $(
            assert!(
                message.contains($needle),
                "{}: the refusal must name {:?}; it said: {message}",
                $context,
                $needle
            );
        )+
        message
    }};

    ($result:expr, $($needle:expr),+ $(,)?) => {{
        let message = match $result {
            Ok(_) => panic!("{} must be refused, but it succeeded", stringify!($result)),
            Err(error) => error.to_string(),
        };
        $(
            assert!(
                message.contains($needle),
                "the refusal must name {:?}; it said: {message}",
                $needle
            );
        )+
        message
    }};
}

#[allow(unused_imports)]
pub(crate) use refuses;

#[allow(dead_code)]
pub mod compare;
#[cfg(feature = "distributed")]
#[allow(dead_code)]
pub mod distributed;
#[allow(dead_code)]
pub mod fragments;
#[allow(dead_code)]
pub mod planner_perf;
#[allow(dead_code)]
pub mod scratch;
#[allow(dead_code)]
pub mod single_phase;
#[allow(dead_code)]
pub mod source_fixture;
#[allow(dead_code)]
pub mod volume;
