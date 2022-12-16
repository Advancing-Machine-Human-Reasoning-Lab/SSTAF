/*
 * Copyright (c) 2022
 * United States Government as represented by the U.S. Army DEVCOM Analysis Center.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mil.sstaftest.maneuver.api;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;
import lombok.extern.jackson.Jacksonized;
import mil.sstaf.core.features.HandlerContent;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
@SuperBuilder
@Jacksonized
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, property = "class")
@EqualsAndHashCode(callSuper = true)
public class ManeuverStateMap extends HandlerContent {

    private final Map<String, ManeuverState> stateMap = new HashMap<>();

    public void addManeuverState(final ManeuverState maneuverState) {
        Objects.requireNonNull(maneuverState, "ManeuverState was null");
        stateMap.put(maneuverState.path, maneuverState);
    }

    public Map<String, ManeuverState> getStateMap() {
        return Collections.unmodifiableMap(stateMap);
    }

}

